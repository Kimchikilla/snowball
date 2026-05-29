"""
multi_agent.py
멀티 에이전트 합의 시스템 (2 체제).

2명의 에이전트가 독립적으로 분석한 뒤, 코드 조율자(Coordinator)가 합의를 도출합니다.
4/26 사고(Bot B 자동 정지) 분석 결과, 4명 트레이더 페르소나 합의는
"보수적 합의 = STOP"으로 끌려가는 편향을 가졌습니다.
이를 "변경 vs 유지"의 의도된 비대칭 합의로 재설계했습니다.

에이전트 구성 (2 체제):
  1. 운영자 (Operator)        — 그리드봇을 실제로 운영하는 관점에서 액션 제안
  2. 비판자 (Exit Critic)     — 액션 변경에 반대하는 변호인. MAINTAIN 옹호.
  + 코드 조율자 (Coordinator) — 두 의견을 규칙으로 종합 (변경에는 만장일치 필요)

모든 에이전트는 "그리드봇 운영자" 관점으로 사고합니다.
일반 트레이더와 달리, 변동성은 위험이 아니라 수익 원천이며,
가격이 그리드 경계 근처에 있는 것은 "위험"이 아니라 "체결 자리"입니다.
"""

import json
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import openai
from google import genai

from config import LLM_PROVIDER, LLM_API_KEY, LLM_MODEL


VALID_ACTIONS = ("MAINTAIN", "WIDEN", "STOP", "SHIFT_UP", "SHIFT_DOWN")


@dataclass
class AgentOpinion:
    role: str
    action: str
    confidence: int      # 1~10
    reason: str


@dataclass
class ConsensusResult:
    final_action: str
    opinions: list        # list[AgentOpinion]
    agreement_rate: float # 0~100%
    reasoning: str


# ─── 그리드봇 헌법 (모든 에이전트 공통 전제) ──────────────
# ① 메커니즘 명시 — 그리드봇은 횡보로 돈 번다는 사실을 시스템 프롬프트 최상단에 박는다.
# ② MAINTAIN 디폴트 — 액션 변경의 입증 책임은 항상 변경 쪽에 있다.
# ③ STOP 재정의 — STOP은 "잠시 멈춤"이 아니라 "그리드 전략 영구 폐기"에 가깝다.

GRID_BOT_CONSTITUTION = """【그리드봇 운영 헌법 — 모든 판단의 전제】

1. 메커니즘:
   - 이 봇은 가격이 위아래로 출렁일 때 매수/매도를 반복해서 돈을 번다.
   - 변동성은 친구다. 횡보는 최적 환경이다.
   - 가격이 그리드 경계 근처에 있다는 것은 "곧 체결될 자리"이며 "위험"이 아니다.
   - 봇을 멈추는 것은 곧 수익 기회를 영구히 포기하는 것이다.

2. 디폴트는 MAINTAIN:
   - 액션을 바꾸려면 명확한 증거가 필요하다. 모호하면 무조건 MAINTAIN.
   - WIDEN/SHIFT: ADX ≥ 25 + 명확한 단방향 추세, 또는 그리드 이탈 24h+ 지속.
   - 1분봉 wick 한 번, 일시적 거래량 spike, 단발성 score 급등은 무시한다.

3. STOP은 거의 없다:
   - STOP은 "잠시 멈춤"이 아니다. 봇 종료 + 자산 청산 + 재시작 시 수수료 발생.
   - 시장이 잠깐 출렁이는 정도로는 절대 STOP하지 마라.
   - STOP 후보 시나리오는 다음과 같이 외부적/시스템적 위험에 한정한다:
     · 1시간 내 ±20% 폭락 같은 시스템적 시장 붕괴
     · 거래소 장애, 디레버리지 이벤트
     · 그리드 전략 자체가 무효한 영구 추세 전환 (ADX 35+ 지속)
   - 위 조건이 아니면 STOP은 답이 아니다. 차라리 SHIFT/WIDEN으로 적응하라.

4. 정지의 비용:
   - 일일 기회비용 = 봇이 돌면 하루에 벌었을 평균 수익.
   - STOP은 자본 보전이 아니라 자본 효율 손실이다.
"""


# ─── 에이전트 프롬프트 ──────────────────────────────────

def _build_market_context(signal, current_price: float,
                          fee_context: str = "") -> str:
    bot_label = getattr(signal, "bot_label", "") or ""
    bot_header = f"[봇: {bot_label}] " if bot_label else ""
    return f"""=== 시장 데이터 ({bot_header}현재가 {current_price:,.0f} USDT) ===
리스크 스코어: {signal.risk_score}/100
상태: {signal.state}

ATR (현재/평균): {signal.atr_current:.1f} / {signal.atr_avg:.1f}
RSI: {signal.rsi:.1f}
볼린저밴드 폭: {signal.bb_width:.1f}%
거래량 배율: {signal.volume_ratio:.1f}x
추세: {signal.trend} (ADX={getattr(signal, 'adx', 0):.1f})
단기 EMA: {getattr(signal, 'ema_short', 0):,.1f}
장기 EMA: {getattr(signal, 'ema_long', 0):,.1f}
{fee_context}"""


AGENT_PROMPTS = {
    "operator": {
        "role": "그리드봇 운영자",
        "system": GRID_BOT_CONSTITUTION + """
당신은 그리드봇 운영자입니다. 시장 데이터(ATR/RSI/BB/거래량/EMA/ADX)와 봇 운영 현황을 보고
이 봇이 지금 어떻게 작동하고 있는지 평가하고, 액션을 제안합니다.

【판단 우선순위】
1. 그리드 안에서 가격이 출렁이는 중? → MAINTAIN. 그게 봇이 돈 버는 자리다.
2. 그리드 이탈 상황이 있나?
   - 24h 미만: MAINTAIN (복귀 대기).
   - 24~48h + ADX≥20: SHIFT 또는 WIDEN.
   - 48h+: WIDEN 강력 권고 (기회비용 > 수수료).
3. ADX < 20 횡보장: 무조건 MAINTAIN.
4. ADX > 25 + 명확한 단방향 EMA 정렬 + 이탈 동반: SHIFT 후보.

【피해야 할 함정】
- "가격이 그리드 상단/하단 80% 도달했으니 위험" — 아니다. 그건 체결 자리다. MAINTAIN.
- "ATR가 평소보다 높으니 STOP" — 아니다. 변동성은 그리드의 친구다.
- "RSI 극단값 한 번" — 무시. 그리드는 mean-reversion 자체로 처리한다.
- "1분봉 wick으로 score 일시 급등" — 단발성 spike는 무조건 MAINTAIN.

【수수료 가드 (액션 변경 시 반드시 확인)】
- 1시간 내 재시작 2회+ → MAINTAIN 강제.
- 예상 재시작 수수료 > 당일 실현 수익의 50% → MAINTAIN.

【STOP은 거의 추천하지 않음】
시스템적 시장 붕괴(±20% 폭락, 거래소 장애, ADX 35+ 영구 추세)가 명확할 때만.
"위험해 보여서", "보수적으로" 같은 정성적 사유로 STOP 권고 금지.

신뢰도 낮으면(1~4) MAINTAIN으로 답하세요. 모호하면 MAINTAIN."""
    },
    "exit_critic": {
        "role": "퇴출 결정 비판자",
        "system": GRID_BOT_CONSTITUTION + """
당신의 유일한 역할은 "액션을 변경하지 말 것"을 강하게 옹호하는 것입니다.
당신은 "그리드봇을 그대로 두는 쪽"의 변호인입니다.

【사고 원칙】
- 변경의 입증 책임은 변경하는 쪽에 있다. 모호하면 MAINTAIN.
- "그냥 두면 어떻게 되는가?"를 항상 먼저 질문한다. 횡보가 이어지면? → 체결 발생 → 수익.
- 액션 변경 = 봇 재시작 = 수수료 발생. 이 비용이 정당화되는가?
- 봇 정지(STOP) = 일일 기회비용 영구 손실. 컨텍스트의 "정지 시 일일 기회비용" 숫자를 본다.

【반박 가이드】
다른 신호가 변경을 시사할 때마다 다음을 자문하라:
- "이 신호가 단발성인가, 지속되는가?" 단발성이면 MAINTAIN.
- "객관적 데이터(이탈 시간, ADX 등)가 변경을 강제하는가, 정성적 인상인가?" 인상이면 MAINTAIN.
- "재시작 수수료보다 변경 후 기대 수익이 명확히 큰가?" 모호하면 MAINTAIN.
- "이 봇이 지금 정상적으로 체결하고 있는가?" 그렇다면 손대지 마라.

【언제 MAINTAIN을 깨는가】
- 그리드 이탈 48h+ + ADX≥20 → WIDEN 동의 가능.
- 거래소 장애 / ±20% 폭락 / ADX 35+ 영구 추세 → STOP 동의 가능 (그러나 매우 드물게).
- 그 외에는 거의 항상 MAINTAIN을 권고하라.

【출력 톤】
당신은 변호인이지 동조자가 아니다. 다른 신호가 약하면 단호하게 MAINTAIN을 권고하라.
액션 변경에 동의할 때도 신뢰도는 보수적으로(7 이하) 매겨라."""
    }
}


# ─── 멀티 에이전트 클래스 ────────────────────────────────

class MultiAgentJudge:
    """2명의 에이전트 의견을 코드 규칙으로 합의하는 의사결정기."""

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "grok": "grok-3-mini",
        "gemini": "gemini-2.5-flash",
    }

    def __init__(self):
        self.available = False
        try:
            self.provider = LLM_PROVIDER.lower()
            self.model = LLM_MODEL or self.DEFAULT_MODELS.get(self.provider, "gpt-4o")

            if not LLM_API_KEY:
                print("[MultiAgent] API 키 미설정 — 멀티 에이전트 비활성화")
                return

            if self.provider == "anthropic":
                self.client = anthropic.Anthropic(api_key=LLM_API_KEY)
            elif self.provider == "openai":
                self.client = openai.OpenAI(api_key=LLM_API_KEY)
            elif self.provider == "grok":
                self.client = openai.OpenAI(
                    api_key=LLM_API_KEY,
                    base_url="https://api.x.ai/v1",
                )
            elif self.provider == "gemini":
                self.client = genai.Client(api_key=LLM_API_KEY)
            else:
                print(f"[MultiAgent] 미지원 provider: {self.provider}")
                return

            self.available = True
        except Exception as e:
            print(f"[MultiAgent] 초기화 실패: {e}")

    def judge(self, signal, current_price: float) -> str:
        """멀티 에이전트 합의 → 최종 액션 반환."""
        if not self.available:
            return "MAINTAIN"

        try:
            result = self._consensus(signal, current_price)
            self._log_result(result)
            return result.final_action
        except Exception as e:
            print(f"[MultiAgent] 합의 실패: {e}")
            return "MAINTAIN"

    def judge_with_detail(self, signal, current_price: float,
                          fee_context: str = "") -> ConsensusResult:
        """상세 합의 결과 반환 (텔레그램 리포트용)."""
        if not self.available:
            return ConsensusResult(
                final_action="MAINTAIN",
                opinions=[],
                agreement_rate=0,
                reasoning="멀티 에이전트 비활성화"
            )
        try:
            return self._consensus(signal, current_price, fee_context)
        except Exception as e:
            return ConsensusResult(
                final_action="MAINTAIN",
                opinions=[],
                agreement_rate=0,
                reasoning=f"합의 실패: {e}"
            )

    # ─── 내부 로직 ───────────────────────────────────────

    def _consensus(self, signal, current_price: float,
                    fee_context: str = "") -> ConsensusResult:
        """에이전트 병렬 호출 → 코드 규칙 합의."""
        context = _build_market_context(signal, current_price, fee_context)

        # 2명 병렬 호출
        opinions = self._gather_opinions(context)

        if not opinions:
            return ConsensusResult("MAINTAIN", [], 0, "에이전트 응답 없음")

        # 조율자 LLM 없이 코드 규칙으로 합의한다.
        final_action, reasoning = self._coordinate(opinions, context)

        # 합의율 계산
        action_counts = {}
        for o in opinions:
            action_counts[o.action] = action_counts.get(o.action, 0) + 1
        max_agree = max(action_counts.values()) if action_counts else 0
        agreement_rate = max_agree / len(opinions) * 100

        return ConsensusResult(
            final_action=final_action,
            opinions=opinions,
            agreement_rate=agreement_rate,
            reasoning=reasoning
        )

    def _gather_opinions(self, context: str) -> list:
        """2명의 에이전트(operator + exit_critic)를 병렬로 호출."""
        opinions = []

        with ThreadPoolExecutor(max_workers=len(AGENT_PROMPTS)) as executor:
            futures = {}
            for agent_id, agent_config in AGENT_PROMPTS.items():
                future = executor.submit(
                    self._ask_agent, agent_id, agent_config, context
                )
                futures[future] = agent_id

            for future in as_completed(futures, timeout=30):
                agent_id = futures[future]
                try:
                    opinion = future.result()
                    if opinion:
                        opinions.append(opinion)
                except Exception as e:
                    print(f"[MultiAgent] {agent_id} 응답 실패: {e}")

        return opinions

    def _ask_agent(self, agent_id: str, config: dict, context: str) -> Optional[AgentOpinion]:
        """개별 에이전트에게 판단 요청."""
        prompt = f"""{context}

다음 액션 중 하나를 선택하고 신뢰도(1~10)와 이유를 답하세요:
- MAINTAIN: 현재 그리드 유지 (디폴트, 모호하면 이걸로)
- WIDEN: 그리드 간격 확대 (재시작 → 수수료 발생)
- SHIFT_UP: 그리드를 위로 이동 (재시작 → 수수료 발생)
- SHIFT_DOWN: 그리드를 아래로 이동 (재시작 → 수수료 발생)
- STOP: 봇 종료 + 자산 청산 (시스템적 시장 붕괴 한정 — 일반적으로 답이 아님)

【판단 가이드라인 — 그리드봇 운영자 관점】
1. 디폴트는 MAINTAIN. 액션 변경에는 명확한 객관적 증거가 필요합니다.
2. 가격이 그리드 경계 80% 도달 → 그건 "체결 자리"이지 위험이 아닙니다. MAINTAIN.
3. 단발성 spike (1분봉 wick, score 일시 급등) → 무시. MAINTAIN.
4. 이탈 상황 (`=== 그리드 이탈 상황 ===`):
   * 이탈 24h 미만: MAINTAIN (복귀 대기).
   * 이탈 24~48h + ADX≥20: WIDEN 또는 SHIFT 권고.
   * 이탈 48h 이상: WIDEN 강력 권고 (복귀 가능성 낮음, 기회비용 > 수수료).
5. 수수료/손익 컨텍스트 확인. 예상 수수료 > 일일 수익의 50% → MAINTAIN.

【STOP 사용 제한】
STOP은 다음 중 하나여야만 후보가 됩니다:
- 1시간 내 시장 ±20% 폭락 같은 시스템적 붕괴.
- ADX 35+ 영구 추세 전환이 명백한 경우.
- 거래소 장애/디레버리지 같은 외부 사건.
"위험해 보여서", "안전하게", "보수적으로" 같은 정성적 사유로 STOP을 권고하지 마세요.
STOP 권고 시 신뢰도는 8 이상이어야 하며, 사유에 위 시스템 위험을 명시해야 합니다.

반드시 아래 JSON 형식으로만 응답:
{{"action": "ACTION", "confidence": 숫자, "reason": "이유 한줄"}}"""

        try:
            raw = self._call_llm(config["system"], prompt)
            return self._parse_opinion(raw, config["role"])
        except Exception as e:
            print(f"[MultiAgent] {config['role']} 호출 오류: {e}")
            return None

    def _coordinate(self, opinions: list, context: str = "") -> tuple:
        """코드 레벨 합의 규칙.

        변경은 두 에이전트가 같은 non-MAINTAIN 액션을 내고 평균 신뢰도가
        7을 초과할 때만 통과한다. 응답 누락, 의견 분산, 낮은 확신은 모두
        MAINTAIN이다.
        """
        expected = len(AGENT_PROMPTS)
        if len(opinions) < expected:
            return (
                "MAINTAIN",
                f"응답 {len(opinions)}/{expected}개만 수신 → 변경 입증 부족",
            )

        actions = [o.action if o.action in VALID_ACTIONS else "MAINTAIN" for o in opinions]
        avg_conf = sum(o.confidence for o in opinions) / len(opinions)

        if any(action == "MAINTAIN" for action in actions):
            return "MAINTAIN", "한 명 이상이 MAINTAIN → 변경 보류"

        if len(set(actions)) != 1:
            return "MAINTAIN", f"의견 분산({', '.join(actions)}) → 변경 보류"

        action = actions[0]
        if avg_conf <= 7:
            return (
                "MAINTAIN",
                f"평균 신뢰도 {avg_conf:.1f}/10 ≤ 7 → 변경 확신 부족",
            )

        if action == "STOP":
            high_conf = all(o.confidence >= 9 for o in opinions)
            systemic = all(self._reason_mentions_systemic_stop(o.reason) for o in opinions)
            if not (high_conf and systemic):
                return (
                    "MAINTAIN",
                    f"STOP 보호 장치 발동 (전원 신뢰도≥9={high_conf}, "
                    f"시스템 위험 명시={systemic}) → MAINTAIN",
                )

        return (
            action,
            f"두 에이전트가 {action}에 일치, 평균 신뢰도 {avg_conf:.1f}/10",
        )

    @staticmethod
    def _reason_mentions_systemic_stop(reason: str) -> bool:
        """STOP 사유가 시스템적 위험인지 보수적으로 확인."""
        text = (reason or "").lower()
        keywords = (
            "시스템적", "시장 붕괴", "폭락", "거래소 장애", "디레버리지",
            "영구 추세", "adx 35", "20%", "exchange outage", "deleveraging",
            "market collapse", "systemic",
        )
        return any(keyword in text for keyword in keywords)

    def _fallback_non_stop(self, opinions: list) -> str:
        """STOP 강등 시 차선 액션 — STOP 제외 다수결, 동률이면 MAINTAIN."""
        non_stop = [o for o in opinions if o.action != "STOP"]
        if not non_stop:
            return "MAINTAIN"
        counts = {}
        for o in non_stop:
            counts[o.action] = counts.get(o.action, 0) + 1
        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]
        # 동률이면 MAINTAIN 우선 (그리드봇 헌법 ②: 모호하면 MAINTAIN)
        priority = {"MAINTAIN": 0, "WIDEN": 1, "SHIFT_UP": 2, "SHIFT_DOWN": 3}
        candidates.sort(key=lambda a: priority.get(a, 99))
        return candidates[0]

    def _majority_vote(self, opinions: list) -> str:
        """폴백: 단순 다수결 (동률 시 MAINTAIN 우선 — 그리드봇 헌법 ②)."""
        if not opinions:
            return "MAINTAIN"

        # 그리드봇 헌법 ②: 디폴트는 MAINTAIN. 동률 시 MAINTAIN 우선.
        # STOP은 만장일치 + 신뢰도 8+ 가드를 _coordinate에서 별도 적용하므로
        # 여기서는 우선순위만 가장 낮게 둔다 (선택돼도 강등됨).
        priority = {"MAINTAIN": 0, "WIDEN": 1, "SHIFT_UP": 2,
                    "SHIFT_DOWN": 3, "STOP": 4}

        counts = {}
        for o in opinions:
            counts[o.action] = counts.get(o.action, 0) + 1

        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]

        # 동률이면 MAINTAIN 우선 (변경의 입증 책임은 변경하는 쪽)
        candidates.sort(key=lambda a: priority.get(a, 99))
        return candidates[0]

    # ─── LLM 호출 ────────────────────────────────────────

    def _call_llm(self, system: str, prompt: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text.strip()
        elif self.provider == "gemini":
            resp = self.client.models.generate_content(
                model=self.model,
                contents=f"{system}\n\n{prompt}",
                config={"max_output_tokens": 200},
            )
            return resp.text.strip()
        else:  # openai, grok (OpenAI 호환)
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content.strip()

    # ─── 파싱 ────────────────────────────────────────────

    def _parse_json(self, raw: str) -> dict:
        """JSON 파싱 (```json``` 래핑 대응)."""
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def _parse_opinion(self, raw: str, role: str) -> Optional[AgentOpinion]:
        """에이전트 응답을 AgentOpinion으로 파싱."""
        try:
            data = self._parse_json(raw)
            action = data.get("action", "MAINTAIN").upper()
            confidence = int(data.get("confidence", 5))
            reason = data.get("reason", "")

            if action not in VALID_ACTIONS:
                action = "MAINTAIN"
            confidence = max(1, min(10, confidence))

            return AgentOpinion(
                role=role,
                action=action,
                confidence=confidence,
                reason=reason
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"[MultiAgent] {role} 응답 파싱 실패: {e}")
            return None

    # ─── 로그 ────────────────────────────────────────────

    def _log_result(self, result: ConsensusResult):
        opinions_str = " | ".join([
            f"{o.role}={o.action}({o.confidence})"
            for o in result.opinions
        ])
        print(
            f"[MultiAgent] 합의={result.final_action} "
            f"(동의율={result.agreement_rate:.0f}%) | {opinions_str}"
        )


def format_consensus_for_telegram(result: ConsensusResult,
                                  bot_label: str = "") -> str:
    """텔레그램 알림용 합의 결과 포맷.

    bot_label이 주어지면 멀티봇 환경에서 어느 봇 합의인지 식별 가능하게
    헤더에 봇 라벨을 표시한다 (Notifier도 메시지 전체에 prefix 추가하지만,
    합의 결과는 핵심 메시지라 헤더에서 한 번 더 노출한다).
    """
    label_part = f" [{bot_label}]" if bot_label else ""
    lines = [
        f"🤖 멀티 에이전트 합의{label_part}",
        f"{'─' * 28}",
        f"최종 결정: {result.final_action}",
        f"합의율: {result.agreement_rate:.0f}%",
        f"{'─' * 28}",
    ]
    for o in result.opinions:
        emoji = {"MAINTAIN": "🟢", "WIDEN": "🟡",
                 "STOP": "🔴", "SHIFT_UP": "⬆️",
                 "SHIFT_DOWN": "⬇️"}.get(o.action, "⚪")
        lines.append(f"{emoji} {o.role}: {o.action} ({o.confidence}/10)")
        lines.append(f"   {o.reason}")
    lines.append(f"{'─' * 28}")
    lines.append(f"📋 {result.reasoning}")
    return "\n".join(lines)
