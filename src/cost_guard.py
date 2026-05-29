"""
cost_guard.py
비용 인식 시스템.

Claude Code 아키텍처에서 영감받은 패턴:
1. 비용 인식 에러 복구 캐스케이드 (무료 → 저비용 → 고비용)
2. 감소 수익 감지 (같은 판단 반복 시 API 호출 스킵)
3. 서킷 브레이커 (연속 실패 시 자동 차단)
4. LLM 응답 캐시 (동일 시장 상황에서 중복 호출 방지)
5. 비용 추적 + 예산 한도
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from collections import deque


# ─── 1. 비용 추적 ────────────────────────────────────────

# 모델별 토큰 단가 (USD per 1M tokens)
MODEL_PRICING = {
    # Anthropic
    "claude-sonnet-4-20250514":  {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":    {"input": 15.0, "output": 75.0},
    "claude-haiku-4-20250414":   {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4o":                    {"input": 2.50, "output": 10.0},
    "gpt-4o-mini":               {"input": 0.15, "output": 0.60},
    "gpt-4.1":                   {"input": 2.0,  "output": 8.0},
    # xAI
    "grok-3-mini":               {"input": 0.30, "output": 0.50},
    "grok-3":                    {"input": 3.0,  "output": 15.0},
    # Google
    "gemini-2.5-flash":          {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro":            {"input": 1.25, "output": 10.0},
    "gemini-2.0-flash":          {"input": 0.10, "output": 0.40},
}

# 호출당 예상 토큰 (입력 ~500, 출력 ~80)
EST_INPUT_TOKENS = 500
EST_OUTPUT_TOKENS = 80


@dataclass
class CostTracker:
    """실시간 API 비용 추적."""
    model: str = ""
    daily_calls: int = 0
    daily_cost_usd: float = 0.0
    total_calls: int = 0
    total_cost_usd: float = 0.0
    max_daily_budget_usd: float = 5.0  # 일일 예산 한도
    _current_date: str = ""

    def _reset_if_new_day(self):
        today = date.today().isoformat()
        if self._current_date != today:
            self.daily_calls = 0
            self.daily_cost_usd = 0.0
            self._current_date = today

    def estimate_call_cost(self, model: str = "", num_calls: int = 1) -> float:
        """호출 예상 비용 산출."""
        m = model or self.model
        pricing = MODEL_PRICING.get(m, {"input": 3.0, "output": 15.0})
        input_cost = (EST_INPUT_TOKENS * pricing["input"]) / 1_000_000
        output_cost = (EST_OUTPUT_TOKENS * pricing["output"]) / 1_000_000
        return (input_cost + output_cost) * num_calls

    def record_call(self, model: str = "", num_calls: int = 1):
        """호출 기록."""
        self._reset_if_new_day()
        cost = self.estimate_call_cost(model, num_calls)
        self.daily_calls += num_calls
        self.daily_cost_usd += cost
        self.total_calls += num_calls
        self.total_cost_usd += cost

    def is_budget_exceeded(self) -> bool:
        """일일 예산 초과 여부."""
        self._reset_if_new_day()
        return self.daily_cost_usd >= self.max_daily_budget_usd

    def budget_remaining(self) -> float:
        self._reset_if_new_day()
        return max(0, self.max_daily_budget_usd - self.daily_cost_usd)

    def summary(self) -> str:
        self._reset_if_new_day()
        return (
            f"오늘: {self.daily_calls}회 ≈ ${self.daily_cost_usd:.4f} "
            f"(한도: ${self.max_daily_budget_usd:.2f} | 잔여: ${self.budget_remaining():.4f})"
        )


# ─── 2. 서킷 브레이커 ────────────────────────────────────

class CircuitBreaker:
    """
    연속 실패 시 자동 차단.

    상태:
      CLOSED  → 정상 (호출 허용)
      OPEN    → 차단 (호출 거부, 쿨다운 대기)
      HALF    → 시험 (1회 허용, 성공 시 CLOSED로 복귀)
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 5, cooldown_sec: int = 300):
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.total_trips = 0  # 총 차단 횟수

    def can_execute(self) -> bool:
        """호출 가능 여부."""
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            # 쿨다운 경과 시 HALF_OPEN으로 전환
            if time.time() - self.last_failure_time >= self.cooldown_sec:
                self.state = self.HALF
                return True
            return False
        # HALF_OPEN: 1회 시험 허용
        return True

    def record_success(self):
        """성공 기록 → CLOSED로 복귀."""
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self):
        """실패 기록 → HALF_OPEN이면 즉시 OPEN, 아니면 임계값 초과 시 OPEN."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == self.HALF:
            # HALF_OPEN에서 실패 → 즉시 다시 OPEN
            self.state = self.OPEN
            self.total_trips += 1
        elif self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            self.total_trips += 1

    def status(self) -> str:
        if self.state == self.OPEN:
            remaining = self.cooldown_sec - (time.time() - self.last_failure_time)
            return f"OPEN (재시도까지 {max(0, remaining):.0f}초)"
        return self.state


# ─── 3. 감소 수익 감지 ───────────────────────────────────

class DiminishingReturnsDetector:
    """
    동일한 판단이 반복되면 LLM 호출을 스킵.

    연속 N회 같은 액션 → "결과가 안 변하니 호출 불필요"
    시장 상황이 변하면 (스코어 변동 > 임계값) 리셋.
    """

    def __init__(self, repeat_threshold: int = 3, score_change_threshold: float = 5.0):
        self.repeat_threshold = repeat_threshold
        self.score_change_threshold = score_change_threshold
        self.recent_actions: deque = deque(maxlen=10)
        self.recent_scores: deque = deque(maxlen=10)
        self.skipped_count: int = 0
        self.total_saved: int = 0

    def should_skip(self, current_score: float) -> bool:
        """LLM 호출을 스킵해야 하는지 판단."""
        # 스코어 이력이 충분하지 않으면 스킵하지 않음
        if len(self.recent_scores) < self.repeat_threshold:
            return False

        # 시장 상황이 변했으면 리셋 (스코어 변동 큰 경우)
        score_delta = abs(current_score - self.recent_scores[-1])
        if score_delta >= self.score_change_threshold:
            self.skipped_count = 0
            return False

        # 최근 N회 액션이 모두 동일하면 스킵
        recent = list(self.recent_actions)[-self.repeat_threshold:]
        if len(recent) == self.repeat_threshold and len(set(recent)) == 1:
            self.skipped_count += 1
            self.total_saved += 1
            return True

        return False

    def record(self, action: str, score: float):
        """판단 결과 기록."""
        self.recent_actions.append(action)
        self.recent_scores.append(score)
        self.skipped_count = 0  # 실제 호출 시 리셋

    def last_action(self) -> Optional[str]:
        """마지막 액션 반환 (스킵 시 재사용)."""
        return self.recent_actions[-1] if self.recent_actions else None


# ─── 4. LLM 응답 캐시 ────────────────────────────────────

class ResponseCache:
    """
    동일 시장 상황에서 중복 LLM 호출 방지.

    시장 상태를 해시하여 캐시 키 생성.
    TTL 이내 동일 조건이면 캐시된 결과 반환.
    """

    def __init__(self, ttl_sec: int = 300, max_size: int = 50):
        self.ttl_sec = ttl_sec
        self.max_size = max_size
        self._cache: dict = {}  # key → (result, timestamp)
        self.hits: int = 0
        self.misses: int = 0

    def _make_key(self, signal) -> str:
        """시장 시그널을 해시 키로 변환."""
        # 스코어를 5점 단위로 양자화 (미세한 차이 무시)
        quantized_score = round(signal.risk_score / 5) * 5
        state = signal.state
        trend = getattr(signal, "trend", "N/A")
        adx_bucket = round(getattr(signal, "trend_strength", 0) / 10) * 10

        raw = f"{quantized_score}|{state}|{trend}|{adx_bucket}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def get(self, signal) -> Optional[str]:
        """캐시 조회. 히트 시 액션 문자열 반환."""
        self._evict_expired()
        key = self._make_key(signal)
        if key in self._cache:
            result, ts = self._cache[key]
            if time.time() - ts < self.ttl_sec:
                self.hits += 1
                return result
            del self._cache[key]
        self.misses += 1
        return None

    def put(self, signal, action: str):
        """캐시 저장."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        key = self._make_key(signal)
        self._cache[key] = (action, time.time())

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self.ttl_sec]
        for k in expired:
            del self._cache[k]

    def _evict_oldest(self):
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0


# ─── 5. 에러 복구 캐스케이드 ─────────────────────────────

class RecoveryCascade:
    """
    비용 인식 에러 복구. 무료 → 저비용 → 고비용 순서.

    Level 0 (무료): 캐시 재사용 / 마지막 액션 반복
    Level 1 (무료): 룰 베이스 폴백 (리스크 스코어 기반)
    Level 2 (저비용): 단일 LLM 호출 (멀티 에이전트 대신)
    Level 3 (고비용): 풀 멀티 에이전트 합의 재시도
    """

    def __init__(self):
        self.current_level = 0
        self.max_level = 3
        self.recovery_count = 0

    def next_strategy(self) -> int:
        """다음 복구 전략 레벨 반환."""
        level = self.current_level
        if self.current_level < self.max_level:
            self.current_level += 1
        self.recovery_count += 1
        return level

    def reset(self):
        """성공 시 리셋."""
        self.current_level = 0

    @staticmethod
    def rule_based_fallback(risk_score: float, trend: str, trend_strength: float) -> str:
        """Level 1: 룰 베이스 폴백 (API 호출 없음)."""
        # LLM이 실패했거나 예산/서킷브레이커로 막힌 상황에서 score만 보고
        # 재시작/정지를 실행하면 단발성 spike에 취약하다. 재배치와 손절은
        # main_agent의 이탈 타이머/손절 코드가 별도로 처리한다.
        return "MAINTAIN"


# ─── 6. 통합 CostGuard ──────────────────────────────────

class CostGuard:
    """모든 비용 인식 컴포넌트를 통합 관리."""

    def __init__(self, model: str = "", daily_budget: float = 5.0):
        self.cost_tracker = CostTracker(model=model, max_daily_budget_usd=daily_budget)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, cooldown_sec=300)
        self.diminishing = DiminishingReturnsDetector(repeat_threshold=3)
        self.cache = ResponseCache(ttl_sec=300, max_size=50)
        self.recovery = RecoveryCascade()

    def pre_check(self, signal) -> tuple:
        """
        LLM 호출 전 체크. (should_call, reason, cached_action)

        Returns:
            (True, "", None) → LLM 호출 진행
            (False, reason, action) → 스킵하고 action 사용
        """
        # 1. 예산 체크
        if self.cost_tracker.is_budget_exceeded():
            fallback = self.recovery.rule_based_fallback(
                signal.risk_score,
                getattr(signal, "trend", "SIDEWAYS"),
                getattr(signal, "trend_strength", 0)
            )
            return False, "일일 예산 초과 → 룰 베이스 폴백", fallback

        # 2. 서킷 브레이커 체크
        if not self.circuit_breaker.can_execute():
            fallback = self.recovery.rule_based_fallback(
                signal.risk_score,
                getattr(signal, "trend", "SIDEWAYS"),
                getattr(signal, "trend_strength", 0)
            )
            return False, f"서킷 브레이커 {self.circuit_breaker.status()} → 룰 베이스 폴백", fallback

        # 3. 캐시 체크
        cached = self.cache.get(signal)
        if cached:
            return False, f"캐시 히트 (히트율={self.cache.hit_rate():.0f}%)", cached

        # 4. 감소 수익 체크
        if self.diminishing.should_skip(signal.risk_score):
            last = self.diminishing.last_action()
            if last:
                return False, f"감소 수익 감지 (연속 동일 판단, 절약={self.diminishing.total_saved}회)", last

        return True, "", None

    def post_success(self, signal, action: str, num_calls: int = 5):
        """LLM 호출 성공 후 기록."""
        self.cost_tracker.record_call(num_calls=num_calls)
        self.circuit_breaker.record_success()
        self.cache.put(signal, action)
        self.diminishing.record(action, signal.risk_score)
        self.recovery.reset()

    def post_failure(self):
        """LLM 호출 실패 후 기록."""
        self.circuit_breaker.record_failure()

    def status_report(self) -> str:
        """상태 리포트."""
        return (
            f"비용: {self.cost_tracker.summary()}\n"
            f"서킷: {self.circuit_breaker.status()} (차단 {self.circuit_breaker.total_trips}회)\n"
            f"캐시: 히트율 {self.cache.hit_rate():.0f}% ({self.cache.hits}/{self.cache.hits + self.cache.misses})\n"
            f"절약: {self.diminishing.total_saved}회 스킵"
        )
