"""
main_agent.py
OKX Adaptive Grid Agent 메인 루프.

실행: python main_agent.py
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx
import anthropic
import openai
from google import genai

import config
from market_analyzer import MarketAnalyzer, MarketSignal
from grid_controller import GridController
from multi_agent import MultiAgentJudge, format_consensus_for_telegram
from cost_guard import CostGuard


# ──────────────────────────────────────────────────────────────
class Notifier:
    """텔레그램 알림 발송 (멀티봇 라벨 prefix + 봇 리스트 footer 지원)."""

    def __init__(self, bot_label: str = "", bot_list_provider=None):
        """
        bot_label: 알림 첫 줄에 prefix할 봇 식별자 (예: "ETH 2000-2500").
        bot_list_provider: 호출하면 OKX 활성 봇 리스트를 footer 문자열로
                           반환하는 callable. None이면 footer 생략.
        """
        # 멀티봇 환경에서 어느 봇의 알림인지 식별하기 위한 prefix.
        self.bot_label: str = bot_label or ""
        # OKX 활성 봇 리스트 footer를 매 알림에 첨부할 provider.
        # 호출 빈도가 잦으니 provider 쪽에서 캐시 권장 (60초 TTL 등).
        self.bot_list_provider = bot_list_provider

    def set_label(self, label: str):
        """봇 라벨 변경 (재시작/SHIFT 등으로 범위가 바뀔 때 호출)."""
        self.bot_label = label or ""

    def _prefix(self, message: str) -> str:
        """메시지 첫 줄에 봇 라벨 prefix를 추가."""
        if not self.bot_label:
            return message
        tag = f"[{self.bot_label}]"
        if message.startswith(tag):
            return message
        return f"{tag} {message}"

    def _append_bot_list(self, message: str) -> str:
        """봇 리스트 footer를 메시지 끝에 첨부.

        provider가 None이거나 빈 문자열을 반환하면 원본 그대로.
        provider 예외는 무시하고 원본 반환 (알림 자체가 막히면 안 됨).
        """
        if not self.bot_list_provider:
            return message
        try:
            footer = self.bot_list_provider()
        except Exception as e:
            print(f"[Notifier] bot list provider 실패: {e}")
            return message
        if not footer:
            return message
        return f"{message}\n{footer}"

    def send(self, message: str):
        message = self._append_bot_list(self._prefix(message))

        # 항상 터미널에도 출력
        ts = datetime.now().strftime("%H:%M:%S")
        DIM = "\033[2m"
        RESET = "\033[0m"
        print(f"{DIM}[{ts}] [TG →]{RESET}")
        for line in message.split("\n"):
            print(f"  {DIM}{line}{RESET}")

        if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        try:
            httpx.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        except httpx.TimeoutException:
            print(f"[Notifier] 텔레그램 발송 타임아웃")
        except Exception as e:
            print(f"[Notifier] 텔레그램 발송 실패: {e}")


# ──────────────────────────────────────────────────────────────
class LLMJudge:
    """리스크 스코어가 애매한 상황에서 LLM에게 판단 요청. (Anthropic / OpenAI 지원)"""

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "grok": "grok-3-mini",
        "gemini": "gemini-2.5-flash",
    }

    def __init__(self):
        self.available = False
        try:
            self.provider = config.LLM_PROVIDER.lower()
            self.model = config.LLM_MODEL or self.DEFAULT_MODELS.get(self.provider, "gpt-4o")

            if not config.LLM_API_KEY:
                print("[LLMJudge] API 키가 설정되지 않음 — LLM 판단 비활성화")
                return

            if self.provider == "anthropic":
                self.client = anthropic.Anthropic(api_key=config.LLM_API_KEY)
            elif self.provider == "openai":
                self.client = openai.OpenAI(api_key=config.LLM_API_KEY)
            elif self.provider == "grok":
                self.client = openai.OpenAI(
                    api_key=config.LLM_API_KEY,
                    base_url="https://api.x.ai/v1",
                )
            elif self.provider == "gemini":
                self.client = genai.Client(api_key=config.LLM_API_KEY)
            else:
                print(f"[LLMJudge] 지원하지 않는 LLM provider: {self.provider} — LLM 판단 비활성화")
                return

            self.available = True
        except Exception as e:
            print(f"[LLMJudge] 초기화 실패: {e} — LLM 판단 비활성화")

    def judge(self, signal: MarketSignal, current_price: float,
              fee_context: str = "") -> str:
        """
        Returns: "MAINTAIN" | "WIDEN" | "SHIFT_UP" | "SHIFT_DOWN" | "STOP"
        """
        if not self.available:
            return "MAINTAIN"

        prompt = f"""
당신은 암호화폐 그리드 거래 전문가입니다.
현재 시장 상황을 분석하고 최적의 행동을 결정해주세요.

=== 현재 상태 ===
리스크 스코어: {signal.risk_score}/100
상태: {signal.state}
현재 가격: {current_price:,.0f} USDT

=== 세부 지표 ===
ATR (현재/평균): {signal.atr_current:.1f} / {signal.atr_avg:.1f}
RSI: {signal.rsi:.1f}
볼린저밴드 폭: {signal.bb_width:.1f}%
거래량 배율: {signal.volume_ratio:.1f}x
{fee_context}
=== 판단 요청 ===
다음 중 하나로만 답하세요 (이유 한 줄 포함):
- MAINTAIN: 현재 그리드를 그대로 유지
- WIDEN: 그리드 간격을 넓혀서 재시작 (수수료 발생!)
- SHIFT_UP: 그리드를 위로 이동 (수수료 발생!)
- SHIFT_DOWN: 그리드를 아래로 이동 (수수료 발생!)
- STOP: 전체 청산 (극단적 상황에서만)

=== 판단 가이드라인 ===
- 컨텍스트에 `=== 그리드 이탈 상황 ===`이 있다면 이탈 기간을 최우선 고려:
  * 이탈 24시간 미만: MAINTAIN 선호 (가격 복귀 대기)
  * 이탈 24~48시간 + 추세 확인 (ADX≥20): WIDEN 또는 SHIFT 권고
  * 이탈 48시간 이상: WIDEN 강력 권고 (복귀 가능성 낮음, 재배치 시급)
- 수수료 가드: 예상 재시작 수수료가 실현 수익의 50%를 초과하면 MAINTAIN
- 이탈 상황이 아니면 리스크 스코어와 추세 기반으로 판단

형식: ACTION|이유
예시: WIDEN|이탈 52시간 초과, 추세 확인됨, 재배치 필요
"""
        try:
            raw = self._call(prompt)
            action = raw.split("|")[0].strip().upper()
            if action not in ("MAINTAIN", "WIDEN", "SHIFT_UP", "SHIFT_DOWN", "STOP"):
                return "MAINTAIN"
            return action
        except Exception as e:
            print(f"[LLMJudge] 오류 ({self.provider}): {e}")
            return "MAINTAIN"

    def _call(self, prompt: str) -> str:
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text.strip()
        elif self.provider == "gemini":
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"max_output_tokens": 100},
            )
            return resp.text.strip()
        else:  # openai, grok (OpenAI 호환)
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────
class OKXDataFetcher:
    """OKX Public API에서 캔들 데이터를 가져옵니다."""

    def __init__(self):
        self.client = httpx.Client(base_url=config.OKX_BASE_URL, timeout=10)

    def get_candles(self) -> list[dict]:
        try:
            resp = self.client.get(
                "/api/v5/market/candles",
                params={"instId": config.SYMBOL, "bar": config.CANDLE_INTERVAL, "limit": config.CANDLE_LOOKBACK}
            )
            data = resp.json().get("data", [])
            if not data:
                print("[OKXDataFetcher] 캔들 데이터가 비어있음")
                return []
            result = []
            for d in reversed(data):
                if not isinstance(d, (list, tuple)) or len(d) < 6:
                    continue
                result.append(
                    {"ts": d[0], "open": d[1], "high": d[2], "low": d[3], "close": d[4], "vol": d[5]}
                )
            return result
        except httpx.TimeoutException:
            print("[OKXDataFetcher] 캔들 요청 타임아웃")
            return []
        except (httpx.HTTPError, ConnectionError) as e:
            print(f"[OKXDataFetcher] 캔들 네트워크 오류: {e}")
            return []
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            print(f"[OKXDataFetcher] 캔들 데이터 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"[OKXDataFetcher] 캔들 조회 실패: {e}")
            return []

    def get_current_price(self) -> Optional[float]:
        try:
            resp = self.client.get(
                "/api/v5/market/ticker",
                params={"instId": config.SYMBOL}
            )
            return float(resp.json()["data"][0]["last"])
        except httpx.TimeoutException:
            print("[OKXDataFetcher] 가격 요청 타임아웃")
            return None
        except (httpx.HTTPError, ConnectionError) as e:
            print(f"[OKXDataFetcher] 가격 네트워크 오류: {e}")
            return None
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            print(f"[OKXDataFetcher] 가격 데이터 파싱 오류: {e}")
            return None
        except Exception as e:
            print(f"[OKXDataFetcher] 가격 조회 실패: {e}")
            return None


# ──────────────────────────────────────────────────────────────
class GridAgent:
    """
    메인 오케스트레이터.

    루프마다:
    1. 시장 데이터 수집
    2. 리스크 스코어 계산
    3. 상태 결정 (NORMAL/CAUTION/WARNING/EMERGENCY)
    4. 액션 실행
    5. 알림 발송
    """

    def __init__(self):
        self.analyzer    = MarketAnalyzer()
        self.controller  = GridController()
        self.fetcher     = OKXDataFetcher()
        # 멀티봇 환경 식별용 라벨 — config.SYMBOL + 그리드 범위로 자동 생성.
        # 텔레그램 채널 하나에 여러 봇이 알림 보내도 [ETH 2000-2400] 같은 prefix로 구분된다.
        self.bot_label   = self._compose_bot_label()
        # 봇 리스트 footer 캐시 (OKX API 빈번 호출 방지, 60초 TTL).
        self._bot_list_cache: tuple[float, str] = (0.0, "")
        self.notifier    = Notifier(
            bot_label=self.bot_label,
            bot_list_provider=self._get_bot_list_footer if config.NOTIFY_INCLUDE_BOT_LIST else None,
        )
        self.llm_judge   = LLMJudge()
        self.multi_agent = MultiAgentJudge()
        self.cost_guard  = CostGuard(model=config.LLM_MODEL, daily_budget=5.0)

        self.prev_state:  str   = "NORMAL"
        self.entry_price: Optional[float] = None   # 첫 진입 가격 (손절 기준)
        self.loop_count:  int   = 0
        self.last_shift_time: Optional[datetime] = None  # 그리드 시프트 쿨다운

        # 체결 감시용: 마지막으로 확인한 체결 ID
        self.last_fill_id: Optional[str] = None
        # 당일 체결 누적 (리포트용)
        self.daily_buys:   int   = 0
        self.daily_sells:  int   = 0
        self.daily_buy_vol:  float = 0.0
        self.daily_sell_vol: float = 0.0
        self.daily_buy_cost: float = 0.0   # 당일 매수 총 비용 (USDT)
        self.daily_sell_revenue: float = 0.0  # 당일 매도 총 수익 (USDT)
        self.daily_fees:     float = 0.0   # 당일 수수료 합계
        # 포지션 추적 (매수 평균가 기반 손익 계산)
        self.holding_qty:    float = 0.0   # 보유 수량
        self.holding_cost:   float = 0.0   # 보유분 총 매수 비용 (USDT)
        self.realized_pnl:   float = 0.0   # 누적 실현 손익
        self.daily_realized: float = 0.0   # 당일 실현 손익
        # 일일 리포트 발송 여부
        self._report_sent_date: Optional[str] = None
        # 그리드 재시작 추적 (수수료 가드용)
        self.grid_restart_times: list = []   # 재시작 시각 기록
        self.grid_restart_count: int = 0     # 당일 재시작 횟수
        self.total_fees_paid: float = 0.0    # 누적 수수료
        # 그리드 이탈 추적
        self.grid_breakout_time: Optional[datetime] = None  # 이탈 시작 시각
        self.grid_breakout_dir: Optional[str] = None        # "ABOVE" | "BELOW"
        self.grid_breakout_notified: bool = False            # 이탈 알림 발송 여부
        self.BREAKOUT_WAIT_SEC: int = config.BREAKOUT_WAIT_HOURS * 3600            # LLM 판단 요청까지 대기
        self.BREAKOUT_HARD_TIMEOUT_SEC: int = config.BREAKOUT_HARD_TIMEOUT_HOURS * 3600  # 강제 SHIFT 한도

        # 날짜 롤오버 감지용
        self._current_date: str = datetime.now().strftime("%Y-%m-%d")

        # 저장된 상태 복원 (파일 없으면 no-op)
        self._load_state()

    @staticmethod
    def _print_disclaimer():
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print()
        print(f"{RED}{'═' * 56}{RESET}")
        print(f"{RED}{BOLD}  ⚠️  투자 위험 경고{RESET}")
        print(f"{RED}{'═' * 56}{RESET}")
        print(f"{RED}  이 소프트웨어는 투자 조언이 아닙니다.{RESET}")
        print(f"{RED}  본 프로그램 사용으로 발생하는 모든 금전적 손실에 대한{RESET}")
        print(f"{RED}  책임은 전적으로 사용자 본인에게 있습니다.{RESET}")
        print(f"{RED}  암호화폐 거래는 원금 손실 위험이 있으며,{RESET}")
        print(f"{RED}  과거 수익이 미래 수익을 보장하지 않습니다.{RESET}")
        print(f"{RED}  반드시 감당 가능한 금액만 투자하세요.{RESET}")
        print(f"{RED}{'═' * 56}{RESET}")
        print()

    def run(self):
        """무한 루프 실행."""
        self._print_disclaimer()
        self._log("🚀 OKX Adaptive Grid Agent 시작")
        self._log(f"   심볼: {config.SYMBOL} | 데모: {config.DEMO_MODE} | 간격: {config.LOOP_INTERVAL_SEC}초")
        self.notifier.send(f"🚀 Grid Agent 시작 | {config.SYMBOL} | Demo={config.DEMO_MODE}")

        # 초기 그리드: 기존 봇 동기화 시도 → 없으면 새로 시작
        resp = self.controller.ensure_grid_running()

        GREEN = "\033[92m"
        CYAN = "\033[96m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        if resp.get("status") == "synced":
            # 기존 봇에 동기화 성공
            print(f"\n{GREEN}{BOLD}{'═' * 56}{RESET}")
            print(f"{GREEN}{BOLD}  ✅ 기존 그리드봇에 연결되었습니다{RESET}")
            print(f"{GREEN}{'═' * 56}{RESET}")
            print(f"  봇 ID    : {resp.get('bot_id')}")
            print(f"  상태     : {resp.get('state')}")
            print(f"  범위     : {resp.get('lower'):,.2f} ~ {resp.get('upper'):,.2f}")
            print(f"  그리드   : {resp.get('grid_num')}개 ({resp.get('mode')})")
            print(f"  투자금   : {resp.get('investment'):,.2f} USDT")
            pnl = resp.get('total_pnl', 0)
            pnl_color = GREEN if pnl >= 0 else RED
            print(f"  현재 손익 : {pnl_color}{pnl:+,.2f} USDT{RESET}")
            print(f"{GREEN}{'─' * 56}{RESET}")
            print(f"  {CYAN}기존 설정에 맞춰 에이전트를 시작합니다.{RESET}\n")

            # Only initialize entry_price when there is no restored state.
            # Risk checks use the live average cost when holdings exist.
            entry = self.fetcher.get_current_price()
            if entry and self.entry_price is None:
                self.entry_price = entry

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            entry_str = f"{self.entry_price:,.2f}" if self.entry_price else "N/A"
            self.notifier.send(
                f"🔄 기존 그리드봇 연결 | {config.SYMBOL}\n"
                f"{'─' * 28}\n"
                f"봇 ID   : {resp.get('bot_id')}\n"
                f"상태    : {resp.get('state')}\n"
                f"모드    : {'Demo (모의거래)' if config.DEMO_MODE else '⚠ Live (실거래)'}\n"
                f"{'─' * 28}\n"
                f"범위    : {resp.get('lower'):,.2f} ~ {resp.get('upper'):,.2f}\n"
                f"그리드  : {resp.get('grid_num')}개 ({resp.get('mode')})\n"
                f"투자금  : {resp.get('investment'):,.2f} USDT\n"
                f"현재가  : {entry_str} USDT\n"
                f"{'─' * 28}\n"
                f"{pnl_emoji} 손익: {pnl:+,.2f} USDT\n"
                f"{'─' * 28}\n"
                f"루프 간격: {config.LOOP_INTERVAL_SEC}초\n"
                f"손절 기준: {config.MAX_LOSS_PERCENT}%"
            )

        elif resp.get("code") == "0":
            # 새 봇 시작 성공
            self._log("✅ 새 그리드봇 시작 성공")
            self.notifier.send(
                f"🚀 새 그리드봇 시작 | {config.SYMBOL}\n"
                f"{'─' * 28}\n"
                f"봇 ID   : {self.controller.bot_id}\n"
                f"모드    : {'Demo (모의거래)' if config.DEMO_MODE else '⚠ Live (실거래)'}\n"
                f"{'─' * 28}\n"
                f"범위    : {config.GRID_LOWER:,.2f} ~ {config.GRID_UPPER:,.2f}\n"
                f"그리드  : {config.GRID_COUNT}개 ({config.GRID_MODE})\n"
                f"예산    : {config.GRID_BUDGET:,.2f} USDT\n"
                f"{'─' * 28}\n"
                f"루프 간격: {config.LOOP_INTERVAL_SEC}초\n"
                f"손절 기준: {config.MAX_LOSS_PERCENT}%"
            )

        else:
            # 시작 실패
            error_msg = ""
            if isinstance(resp.get("data"), list) and resp["data"]:
                error_msg = resp["data"][0].get("sMsg", "")

            print(f"\n{RED}{BOLD}{'═' * 56}{RESET}")
            print(f"{RED}{BOLD}  ❌ 그리드봇 시작 실패{RESET}")
            print(f"{RED}{'═' * 56}{RESET}")

            if "Insufficient balance" in str(error_msg):
                print(f"{RED}  원인: 잔고 부족 (Insufficient balance){RESET}")
                print(f"{RED}  현재 설정 예산: {config.GRID_BUDGET} USDT{RESET}")
                print()
                print(f"  💡 해결 방법:")
                if config.DEMO_MODE:
                    print(f"     1. OKX 데모 계정에 충분한 USDT를 충전하세요")
                    print(f"        (okx.com → 데모 트레이딩 → 자산 → 충전)")
                    print(f"     2. 또는 설정에서 그리드 예산을 줄여주세요")
                else:
                    print(f"     1. OKX 계정에 충분한 USDT를 입금하세요")
                    print(f"     2. 또는 설정에서 그리드 예산을 줄여주세요")
            else:
                print(f"{RED}  에러: {error_msg or resp}{RESET}")

            print(f"{RED}{'═' * 56}{RESET}")
            print(f"\n  프로그램을 종료합니다. 문제를 해결한 후 다시 실행해주세요.\n")
            self.notifier.send(f"❌ 그리드봇 시작 실패: {error_msg}")
            sys.exit(1)

        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                self._log("사용자 중단 요청")
                self.notifier.send("⛔ Grid Agent 수동 종료")
                break
            except SystemExit:
                self._log("시스템 종료 요청")
                self.notifier.send("⛔ Grid Agent 시스템 종료")
                break
            except Exception as e:
                self._log(f"루프 오류: {e}", level="ERROR")
                try:
                    self.notifier.send(f"❌ Agent 오류 발생 (상세 내용은 터미널 확인)")
                except Exception:
                    pass

            try:
                self._wait_with_progress(config.LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                self._log("사용자 중단 요청 (대기 중)")
                self.notifier.send("⛔ Grid Agent 수동 종료")
                break

    # ─── 단일 루프 ─────────────────────────────────────────

    def _tick(self):
        self.loop_count += 1
        ts = datetime.now().strftime("%H:%M:%S")
        DIM = "\033[2m"
        RESET = "\033[0m"
        CYAN = "\033[96m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        MAGENTA = "\033[95m"
        BOLD = "\033[1m"

        # 자정 기준 일일 카운터 리셋
        today = datetime.now().strftime("%Y-%m-%d")
        if not hasattr(self, "_current_date"):
            self._current_date = today
        if self._current_date != today:
            self._current_date = today
            self.daily_buys = 0
            self.daily_sells = 0
            self.daily_buy_vol = 0.0
            self.daily_sell_vol = 0.0
            self.daily_buy_cost = 0.0
            self.daily_sell_revenue = 0.0
            self.daily_fees = 0.0
            self.daily_realized = 0.0
            self.grid_restart_count = 0
            self._log("📅 날짜 변경 — 일일 카운터 리셋")

        print()
        print(f"{CYAN}{BOLD}{'═' * 60}{RESET}")
        print(f"{CYAN}{BOLD}  TICK #{self.loop_count}  [{ts}]  {config.SYMBOL}{RESET}")
        print(f"{CYAN}{'═' * 60}{RESET}")

        # 1. 데이터 수집
        print(f"\n{DIM}[1/10]{RESET} {BOLD}데이터 수집{RESET} ─ OKX API 호출 중...")
        try:
            candles = self.fetcher.get_candles()
            price   = self.fetcher.get_current_price()
        except Exception as e:
            print(f"  {RED}✗ 실패: {e}{RESET}")
            return

        if price is None:
            print(f"  {RED}✗ 현재 가격 조회 불가 — 스킵{RESET}")
            return
        if not candles:
            print(f"  {RED}✗ 캔들 데이터 없음 — 스킵{RESET}")
            return
        print(f"  {GREEN}✓{RESET} 현재가: {BOLD}{price:,.0f} USDT{RESET} | 캔들: {len(candles)}개")

        # 2. 리스크 분석
        print(f"\n{DIM}[2/10]{RESET} {BOLD}리스크 분석{RESET} ─ ATR / RSI / BB / Volume / EMA / ADX")
        try:
            signal = self.analyzer.analyze(candles)
        except Exception as e:
            print(f"  {RED}✗ 분석 실패: {e}{RESET}")
            return

        trend = getattr(signal, "trend", "N/A")
        trend_strength = getattr(signal, "trend_strength", 0.0)
        ema_s = getattr(signal, "ema_short", 0)
        ema_l = getattr(signal, "ema_long", 0)
        state_emoji = {"NORMAL": "🟢", "CAUTION": "🟡", "WARNING": "🟠", "EMERGENCY": "🔴"}
        emoji = state_emoji.get(signal.state, "⚪")

        print(f"  ┌────────────────────────────────────────────┐")
        print(f"  │ ATR  = {signal.atr_score:>5.1f}/30  (현재={signal.atr_current:.1f} 평균={signal.atr_avg:.1f})")
        print(f"  │ RSI  = {signal.rsi_score:>5.1f}/25  (RSI={signal.rsi:.1f})")
        print(f"  │ BB   = {signal.bb_score:>5.1f}/25  (폭={signal.bb_width:.2f}%)")
        print(f"  │ Vol  = {signal.volume_score:>5.1f}/20  (배율={signal.volume_ratio:.1f}x)")
        print(f"  ├────────────────────────────────────────────┤")

        trend_color = GREEN if trend == "BULLISH" else RED if trend == "BEARISH" else YELLOW
        print(f"  │ 추세 = {trend_color}{BOLD}{trend}{RESET}  (ADX={trend_strength:.1f})")
        print(f"  │ EMA  = 단기 {ema_s:,.1f} / 장기 {ema_l:,.1f}")
        print(f"  ├────────────────────────────────────────────┤")

        score_color = GREEN if signal.risk_score <= 30 else YELLOW if signal.risk_score <= 60 else RED
        print(f"  │ {BOLD}총점 = {score_color}{signal.risk_score:.1f}/100{RESET}  →  {emoji} {signal.state}")
        print(f"  └────────────────────────────────────────────┘")

        # 3. 손절 체크
        print(f"\n{DIM}[3/10]{RESET} {BOLD}손절 조건 체크{RESET} ─ 평균단/총손익 기준 {config.MAX_LOSS_PERCENT}% 이상 손실?")
        try:
            stop_status = self._stop_loss_status(price)
            if stop_status["triggered"]:
                print(f"  {RED}{BOLD}✗ 손절 조건 도달! 긴급 청산 실행{RESET}")
                self.controller.emergency_stop()
                self.notifier.send(
                    f"💀 손절 청산 | {config.SYMBOL} | 현재가={price:,.0f}\n"
                    f"사유: {stop_status['reason']}"
                )
                return
        except Exception as e:
            print(f"  {RED}✗ 체크 실패: {e}{RESET}")
            stop_status = None

        if stop_status and stop_status["basis_price"] > 0:
            print(
                f"  {GREEN}✓{RESET} 기준={stop_status['basis_price']:,.0f} "
                f"({stop_status['basis']}) | 가격손익={-stop_status['price_loss_pct']:+.2f}% | "
                f"총손익={stop_status['total_pnl']:+,.2f} "
                f"({stop_status['total_pnl_pct']:+.2f}%) | 한도={config.MAX_LOSS_PERCENT}%"
            )
        else:
            print(f"  {GREEN}✓{RESET} 정상 (진입가 미설정)")

        # 4. 체결 감시
        print(f"\n{DIM}[4/10]{RESET} {BOLD}체결 내역 감시{RESET} ─ 신규 매수/매도 확인")
        try:
            self._check_fills(price)
            # 미실현 손익 계산
            unrealized = 0.0
            if self.holding_qty > 0 and price:
                avg_buy = self.holding_cost / self.holding_qty
                unrealized = (price - avg_buy) * self.holding_qty
            total_day = self.daily_realized + unrealized
            pnl_c = GREEN if total_day >= 0 else RED
            print(f"  {GREEN}✓{RESET} 매수={self.daily_buys} 매도={self.daily_sells} | "
                  f"보유={self.holding_qty:.6f} | "
                  f"실현={self.daily_realized:+,.4f} | "
                  f"미실현={unrealized:+,.4f} | "
                  f"{pnl_c}합계={total_day:+,.4f}{RESET}")
        except Exception as e:
            print(f"  {RED}✗ 감시 실패: {e}{RESET}")

        # 4.5 그리드 이탈 체크
        breakout_action = self._check_grid_breakout(signal, price)
        if breakout_action is not None:
            gl = self.controller.current_lower
            gu = self.controller.current_upper
            elapsed_str = ""
            if self.grid_breakout_time:
                elapsed = (datetime.now() - self.grid_breakout_time).total_seconds()
                elapsed_str = f" | 이탈 {elapsed/60:.0f}분"
            print(f"  {YELLOW}⚠ 그리드 이탈 감지{RESET} | "
                  f"범위: {gl:,.2f}~{gu:,.2f} | "
                  f"현재가: {price:,.2f}"
                  f"{elapsed_str} → {breakout_action}")
            action = breakout_action
            # 바로 실행으로 점프
            action_colors = {
                "MAINTAIN": GREEN, "WIDEN": YELLOW,
                "STOP": RED, "SHIFT_UP": CYAN, "SHIFT_DOWN": CYAN
            }
            ac = action_colors.get(action, RESET)
            print(f"\n{DIM}[6/10]{RESET} {BOLD}액션 실행{RESET} ─ {ac}{action}{RESET}")
            try:
                self._execute(action, signal, price)
                print(f"  {GREEN}✓{RESET} 실행 완료")
            except Exception as e:
                print(f"  {RED}✗ 실행 실패: {e}{RESET}")
            # 나머지 스텝 계속 (리포트, 상태변화 등)
            self._post_action_steps(signal, price, action, trend, trend_strength,
                                     emoji, score_color, trend_color, ac,
                                     DIM, RESET, BOLD, GREEN, RED, YELLOW, CYAN)
            return

        # 5. 의사결정
        print(f"\n{DIM}[5/10]{RESET} {BOLD}의사결정{RESET} ─ 추세 판단 → 리스크 스코어 → 에이전트 합의")
        action = "MAINTAIN"
        try:
            action = self._decide_action(signal, price)
        except Exception as e:
            print(f"  {RED}✗ 결정 실패, MAINTAIN 유지: {e}{RESET}")

        action_colors = {
            "MAINTAIN": GREEN, "WIDEN": YELLOW,
            "STOP": RED, "SHIFT_UP": CYAN, "SHIFT_DOWN": CYAN
        }
        ac = action_colors.get(action, RESET)
        print(f"  {BOLD}→ 결정: {ac}{action}{RESET}")

        # 6. 액션 실행
        print(f"\n{DIM}[6/10]{RESET} {BOLD}액션 실행{RESET} ─ {ac}{action}{RESET}")
        try:
            self._execute(action, signal, price)
            print(f"  {GREEN}✓{RESET} 실행 완료")
        except Exception as e:
            print(f"  {RED}✗ 실행 실패: {e}{RESET}")

        self._post_action_steps(signal, price, action, trend, trend_strength,
                                 emoji, score_color, trend_color, ac,
                                 DIM, RESET, BOLD, GREEN, RED, YELLOW, CYAN)

    def _post_action_steps(self, signal, price, action, trend, trend_strength,
                            emoji, score_color, trend_color, ac,
                            DIM, RESET, BOLD, GREEN, RED, YELLOW, CYAN):
        """틱의 7~10 스텝 (일일 리포트, 상태 변화, 비용, 요약)."""
        # 7. 일일 리포트
        print(f"\n{DIM}[7/10]{RESET} {BOLD}일일 리포트 체크{RESET} ─ {config.DAILY_REPORT_HOUR}시 발송")
        try:
            self._check_daily_report(price)
            sent = "발송됨" if self._report_sent_date == datetime.now().strftime("%Y-%m-%d") else "미발송"
            print(f"  {GREEN}✓{RESET} {sent}")
        except Exception as e:
            print(f"  {RED}✗ 실패: {e}{RESET}")

        # 8. 상태 변화 알림
        print(f"\n{DIM}[8/10]{RESET} {BOLD}상태 변화 감지{RESET} ─ {self.prev_state} → {signal.state}")
        try:
            if signal.state != self.prev_state:
                if signal.state in config.NOTIFY_ON_STATES:
                    self.notifier.send(
                        f"{emoji} 상태 변화: {self.prev_state} → {signal.state}\n"
                        f"리스크 점수: {signal.risk_score}/100\n"
                        f"추세: {trend}(ADX={trend_strength:.1f})\n"
                        f"{signal.reason}\n"
                        f"현재가: {price:,.0f}\n"
                        f"액션: {action}"
                    )
                    print(f"  {YELLOW}⚡ 상태 변화 알림 발송: {self.prev_state} → {signal.state}{RESET}")
                else:
                    print(f"  {DIM}상태 변화 (알림 대상 아님): {self.prev_state} → {signal.state}{RESET}")
                self.prev_state = signal.state
            else:
                print(f"  {DIM}변화 없음 ({signal.state}){RESET}")
        except Exception as e:
            print(f"  {RED}✗ 알림 실패: {e}{RESET}")

        # 9. 비용 현황
        print(f"\n{DIM}[9/10]{RESET} {BOLD}비용 현황{RESET}")
        for line in self.cost_guard.status_report().split("\n"):
            print(f"  {DIM}{line}{RESET}")

        # 10. 요약 + 텔레그램 틱 리포트
        print(f"\n{DIM}[10/10]{RESET} {BOLD}틱 완료{RESET}")
        summary_line = (
            f"{emoji} {signal.state} | {score_color}{signal.risk_score:.1f}/100{RESET} | "
            f"{trend_color}{trend}(ADX={trend_strength:.1f}){RESET} | "
            f"{ac}{action}{RESET} | {price:,.0f} USDT"
        )
        print(f"  {summary_line}")

        # 매 틱 텔레그램 발송
        self._send_tick_report(signal, price, action, trend, trend_strength)

        # 상태 저장 (재시작 시 이어받기)
        self._save_state()

        print(f"{CYAN}{'─' * 60}{RESET}")

    # ─── 의사결정 ──────────────────────────────────────────

    def _detect_events(self, signal: MarketSignal, price: float) -> list[str]:
        """이벤트 감지: 에이전트 호출이 필요한 상황인지 판별."""
        events = []
        trigger_score = float(getattr(config, "LLM_TRIGGER_SCORE", 55) or 55)
        trend = getattr(signal, "trend", "SIDEWAYS")
        trend_strength = float(getattr(signal, "trend_strength", 0.0) or 0.0)
        atr_spike = (
            signal.atr_avg > 0
            and signal.atr_current >= signal.atr_avg * 3
        )
        volume_spike = signal.volume_ratio >= 5.0
        strong_trend = trend != "SIDEWAYS" and trend_strength >= 25

        # 그리드 경계 80% 도달은 체결 자리라 LLM 호출 사유가 아니다.
        # 실제 범위 이탈은 _check_grid_breakout()에서 별도로 다룬다.
        if signal.risk_score >= trigger_score:
            events.append(
                f"리스크 스코어 {signal.risk_score:.1f}/100 "
                f"(호출 기준 {trigger_score:.0f})"
            )
            if atr_spike:
                events.append(
                    f"ATR 급등 ({signal.atr_current:.1f} / 평균 "
                    f"{signal.atr_avg:.1f} = {signal.atr_current/signal.atr_avg:.1f}배)"
                )
            if volume_spike:
                events.append(f"거래량 폭발 ({signal.volume_ratio:.1f}x)")
            if strong_trend:
                events.append(f"강한 {trend} 추세 (ADX={trend_strength:.1f})")
        elif atr_spike and strong_trend:
            events.append(
                f"ATR 급등 + 강한 {trend} 추세 "
                f"(ATR {signal.atr_current/signal.atr_avg:.1f}배, ADX={trend_strength:.1f})"
            )
        elif volume_spike and strong_trend:
            events.append(
                f"거래량 폭발 + 강한 {trend} 추세 "
                f"(Vol={signal.volume_ratio:.1f}x, ADX={trend_strength:.1f})"
            )

        return events

    def _decide_action(self, signal: MarketSignal, price: float) -> str:
        """이벤트 기반 의사결정. 이벤트가 있을 때만 에이전트 호출."""

        # 멀티봇 환경 대비: 라벨을 signal에 주입해 LLM 컨텍스트에 봇 식별자 노출.
        self._attach_bot_label(signal)

        score = signal.risk_score
        trend = getattr(signal, "trend", "SIDEWAYS")
        trend_strength = getattr(signal, "trend_strength", 0.0)

        # 이벤트 감지
        events = self._detect_events(signal, price)

        # 이벤트 없으면 에이전트 호출 안 함 → MAINTAIN
        if not events:
            return "MAINTAIN"

        # 이벤트 발생 → 에이전트 호출
        event_str = "\n".join(f"  - {e}" for e in events)
        self._log(f"⚡ 이벤트 감지 ({len(events)}건):\n{event_str}")

        # ─── 그리드봇 헌법 ③: score 단발 spike로 코드가 STOP을 결정하지 않는다.
        # 극단적 score여도 멀티 에이전트 합의 + 코드 레벨 STOP 가드를 거치도록 한다.
        # (이전: score≥90 → 즉시 STOP. 단발성 spike에 봇이 멈추는 사고 원인이었음.)

        # CostGuard 체크
        should_call, reason, cached_action = self.cost_guard.pre_check(signal)
        if not should_call:
            self._log(f"💰 CostGuard 스킵: {reason} → {cached_action}")
            return cached_action

        # 에이전트에게 이벤트 컨텍스트 포함해서 판단 요청
        fee_ctx = self._build_fee_context(price)
        event_ctx = f"\n=== 이벤트 (에이전트 호출 사유) ===\n{event_str}\n"

        # 실제 LLM 호출
        combined_ctx = fee_ctx + event_ctx
        try:
            if config.MULTI_AGENT_MODE and self.multi_agent.available:
                result = self.multi_agent.judge_with_detail(
                    signal, price, fee_context=combined_ctx
                )
                self._log(
                    f"멀티 에이전트 합의: {result.final_action} "
                    f"(동의율={result.agreement_rate:.0f}%, score={score})"
                )
                self.notifier.send(format_consensus_for_telegram(result, bot_label=self.bot_label))
                # 2명 에이전트만 호출. 합의는 코드 규칙으로 처리한다.
                self.cost_guard.post_success(signal, result.final_action, num_calls=2)
                action = result.final_action
            else:
                action = self.llm_judge.judge(
                    signal, price, fee_context=combined_ctx
                )
                self._log(f"LLM 단독 판단: {action} (score={score})")
                self.cost_guard.post_success(signal, action, num_calls=1)
        except Exception as e:
            self._log(f"LLM 호출 실패: {e} → 룰 베이스 폴백", level="ERROR")
            self.cost_guard.post_failure()
            action = self.cost_guard.recovery.rule_based_fallback(
                score, trend, trend_strength
            )
            self._log(f"폴백 결정: {action}")

        # 수수료 가드: 재시작 액션이면 체크
        if action in ("WIDEN", "SHIFT_UP", "SHIFT_DOWN"):
            allowed, skip_reason = self._check_restart_allowed(action, price)
            if not allowed:
                self._log(f"🛡️ 수수료 가드: {skip_reason} → MAINTAIN 유지")
                self.notifier.send(
                    f"🛡️ 그리드 조정 스킵\n"
                    f"에이전트 판단: {action}\n"
                    f"사유: {skip_reason}\n"
                    f"→ MAINTAIN 유지"
                )
                return "MAINTAIN"

        return action

    def _execute(self, action: str, signal: MarketSignal, price: float):
        """액션을 실제 API 호출로 변환."""

        if action == "MAINTAIN":
            self.controller.ensure_grid_running()

        elif action == "WIDEN":
            self._record_grid_restart()
            old_lower = self.controller.current_lower
            old_upper = self.controller.current_upper
            self.controller.widen_grid(
                atr_value=signal.atr_current,
                current_price=price
            )
            new_lower = self.controller.current_lower
            new_upper = self.controller.current_upper
            # 그리드 범위 변경 → 라벨 갱신 (멀티봇 알림 식별 정확도 유지)
            self._refresh_bot_label()
            est_fee = self.holding_qty * price * 0.002 if self.holding_qty > 0 else 0
            self.notifier.send(
                f"🔄 그리드 확대 (WIDEN) | {config.SYMBOL}\n"
                f"{'─' * 28}\n"
                f"이전 범위: {old_lower:,.2f} ~ {old_upper:,.2f}\n"
                f"새 범위  : {new_lower:,.2f} ~ {new_upper:,.2f}\n"
                f"현재가   : {price:,.0f} USDT\n"
                f"ATR      : {signal.atr_current:.1f}\n"
                f"{'─' * 28}\n"
                f"예상 수수료: ~{est_fee:,.4f} USDT\n"
                f"당일 누적 수수료: {self.daily_fees:,.4f} USDT\n"
                f"당일 재시작: {self.grid_restart_count}회"
            )

        elif action == "SHIFT_UP":
            self._record_grid_restart()
            try:
                grid_lower = getattr(self.controller, "current_lower", None)
                grid_upper = getattr(self.controller, "current_upper", None)
                if grid_lower is not None and grid_upper is not None:
                    old_lower, old_upper = grid_lower, grid_upper
                    grid_range = grid_upper - grid_lower
                    offset = grid_range * 0.1
                    new_center = price + offset
                    self.controller.shift_grid_center(new_center, price)
                    self._refresh_bot_label()
                    self.last_shift_time = datetime.now()
                    trend_strength = getattr(signal, "trend_strength", 0.0)
                    est_fee = self.holding_qty * price * 0.002 if self.holding_qty > 0 else 0
                    self.notifier.send(
                        f"📈 그리드 상향 시프트 | {config.SYMBOL}\n"
                        f"{'─' * 28}\n"
                        f"이전 범위: {old_lower:,.2f} ~ {old_upper:,.2f}\n"
                        f"새 범위  : {self.controller.current_lower:,.2f} ~ {self.controller.current_upper:,.2f}\n"
                        f"새 중심  : {new_center:,.0f} USDT\n"
                        f"현재가   : {price:,.0f} USDT\n"
                        f"추세: BULLISH (ADX={trend_strength:.1f})\n"
                        f"{'─' * 28}\n"
                        f"예상 수수료: ~{est_fee:,.4f} USDT\n"
                        f"당일 누적 수수료: {self.daily_fees:,.4f} USDT\n"
                        f"당일 재시작: {self.grid_restart_count}회"
                    )
            except Exception as e:
                self._log(f"SHIFT_UP 실행 실패: {e}", level="ERROR")

        elif action == "SHIFT_DOWN":
            self._record_grid_restart()
            try:
                grid_lower = getattr(self.controller, "current_lower", None)
                grid_upper = getattr(self.controller, "current_upper", None)
                if grid_lower is not None and grid_upper is not None:
                    old_lower, old_upper = grid_lower, grid_upper
                    grid_range = grid_upper - grid_lower
                    offset = grid_range * 0.1
                    new_center = price - offset
                    self.controller.shift_grid_center(new_center, price)
                    self._refresh_bot_label()
                    self.last_shift_time = datetime.now()
                    trend_strength = getattr(signal, "trend_strength", 0.0)
                    est_fee = self.holding_qty * price * 0.002 if self.holding_qty > 0 else 0
                    self.notifier.send(
                        f"📉 그리드 하향 시프트 | {config.SYMBOL}\n"
                        f"{'─' * 28}\n"
                        f"이전 범위: {old_lower:,.2f} ~ {old_upper:,.2f}\n"
                        f"새 범위  : {self.controller.current_lower:,.2f} ~ {self.controller.current_upper:,.2f}\n"
                        f"새 중심  : {new_center:,.0f} USDT\n"
                        f"현재가   : {price:,.0f} USDT\n"
                        f"추세: BEARISH (ADX={trend_strength:.1f})\n"
                        f"{'─' * 28}\n"
                        f"예상 수수료: ~{est_fee:,.4f} USDT\n"
                        f"당일 누적 수수료: {self.daily_fees:,.4f} USDT\n"
                        f"당일 재시작: {self.grid_restart_count}회"
                    )
            except Exception as e:
                self._log(f"SHIFT_DOWN 실행 실패: {e}", level="ERROR")

        elif action == "STOP":
            self.controller.emergency_stop()
            self.notifier.send(
                f"🔴 긴급 청산 완료 | {config.SYMBOL}\n"
                f"리스크 점수: {signal.risk_score}/100\n"
                f"사유: {signal.reason}"
            )

    # ─── 체결 감시 ─────────────────────────────────────────

    def _check_fills(self, current_price: float):
        """그리드봇 체결 내역을 감지하고 텔레그램으로 알림."""
        if not self.controller.bot_id:
            return
        try:
            resp = self.controller._get(
                "/api/v5/tradingBot/grid/sub-orders",
                params={
                    "algoId": self.controller.bot_id,
                    "algoOrdType": "grid",
                    "type": "filled",
                }
            )
            fills = resp.get("data", [])
        except Exception as e:
            self._log(f"체결 내역 조회 실패: {e}", level="ERROR")
            return

        if not fills or not isinstance(fills, list):
            return

        # 첫 실행 시 마지막 ID만 기록
        if self.last_fill_id is None:
            self.last_fill_id = fills[0].get("ordId", "") if isinstance(fills[0], dict) else ""
            return

        # 새 체결만 필터링 (최신순으로 오므로 last_fill_id 이전까지)
        new_fills = []
        for f in fills:
            if not isinstance(f, dict):
                continue
            if f.get("ordId", "") == self.last_fill_id:
                break
            new_fills.append(f)

        if not new_fills:
            return

        self.last_fill_id = new_fills[0].get("ordId", "")

        for f in reversed(new_fills):
            try:
                side = f.get("side", "")
                px   = float(f.get("px", 0))
                sz   = float(f.get("sz", 0))
                fee  = float(f.get("fee", 0))
            except (ValueError, TypeError) as e:
                self._log(f"체결 데이터 파싱 오류: {e}", level="ERROR")
                continue

            cost = px * sz
            coin = config.SYMBOL.split("-")[0]
            # 수수료 USDT 환산 (매수=코인으로 차감 → 현재가 곱, 매도=USDT)
            fee_usdt = abs(fee) * current_price if side == "buy" else abs(fee)
            self.daily_fees += fee_usdt
            self.total_fees_paid += fee_usdt

            if side == "buy":
                emoji = "🟢"
                label = "매수"
                self.daily_buys += 1
                self.daily_buy_vol += sz
                self.daily_buy_cost += cost
                self.holding_qty += sz
                # Include buy fees in cost basis so unrealized/realized PnL
                # reflects the actual cost of building inventory.
                self.holding_cost += cost + fee_usdt
                avg_price = self.holding_cost / self.holding_qty if self.holding_qty > 0 else px
                pnl_line = f"평균 매수가: {avg_price:,.2f} USDT"
            else:
                emoji = "🔴"
                label = "매도"
                self.daily_sells += 1
                self.daily_sell_vol += sz
                self.daily_sell_revenue += cost
                if self.holding_qty > 0:
                    avg_buy = self.holding_cost / self.holding_qty
                    profit = (px - avg_buy) * sz - fee_usdt
                    self.realized_pnl += profit
                    self.daily_realized += profit
                    sell_ratio = min(sz / self.holding_qty, 1.0)
                    self.holding_cost -= self.holding_cost * sell_ratio
                    self.holding_qty = max(self.holding_qty - sz, 0.0)
                    profit_emoji = "💰" if profit >= 0 else "💸"
                    pnl_line = f"{profit_emoji} 실현 손익: {profit:+,.4f} USDT"
                else:
                    pnl_line = ""

            diff = current_price - px
            diff_pct = diff / px * 100 if px > 0 else 0
            diff_emoji = "🔺" if diff > 0 else "🔻" if diff < 0 else "▪️"

            msg = (
                f"{emoji} {label} 체결 | {config.SYMBOL}\n"
                f"{'━' * 28}\n"
                f"💵 {label} 가격 : {px:,.0f} USDT\n"
                f"📦 수량      : {sz:.6f} {coin}\n"
                f"💲 금액      : {cost:,.2f} USDT\n"
                f"🏷️ 수수료    : {abs(fee):.6f}\n"
                f"{'━' * 28}\n"
                f"{pnl_line}\n"
                f"📊 현재가: {current_price:,.0f} USDT "
                f"({diff_emoji} {diff:+,.0f})"
            )
            self.notifier.send(msg)
            self._log(f"{emoji} {label} 체결 | {px:,.0f} × {sz:.6f} = {cost:,.2f} USDT")

    # ─── 일일 리포트 ─────────────────────────────────────

    def _check_daily_report(self, current_price: float):
        """매일 지정 시간에 당일 손익 리포트를 텔레그램으로 발송."""
        try:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # 이미 오늘 보냈으면 스킵
            if self._report_sent_date == today:
                return

            # 지정 시간이 안 됐으면 스킵
            if now.hour < config.DAILY_REPORT_HOUR:
                return

            # 날짜가 바뀌었는지 체크 (리포트 발송 후 리셋)
            is_new_day = self._report_sent_date and self._report_sent_date != today

            # PnL 조회
            pnl_available = True
            try:
                pnl = self.controller.get_grid_pnl()
                grid_profit = pnl.get("grid_profit", 0)
                float_profit = pnl.get("float_profit", 0)
                total_pnl = pnl.get("total_pnl", 0)
                investment = pnl.get("investment", 0)
                roi = (total_pnl / investment * 100) if investment > 0 else 0
            except Exception as e:
                self._log(f"PnL 조회 실패: {e}", level="ERROR")
                pnl_available = False

            if pnl_available:
                pnl_emoji = "📈" if total_pnl >= 0 else "📉"
                pnl_section = (
                    f"{pnl_emoji} 손익 현황\n"
                    f"  그리드 수익: {grid_profit:+,.2f} USDT\n"
                    f"  평가 손익: {float_profit:+,.2f} USDT\n"
                    f"  총 손익: {total_pnl:+,.2f} USDT\n"
                    f"  수익률: {roi:+.2f}%"
                )
            else:
                pnl_section = "⚠️ 손익 현황: 조회 실패"

            msg = (
                f"📊 일일 리포트 | {today}\n"
                f"{'─' * 28}\n"
                f"심볼: {config.SYMBOL}\n"
                f"현재가: {current_price:,.0f} USDT\n"
                f"{'─' * 28}\n"
                f"{pnl_section}\n"
                f"{'─' * 28}\n"
                f"📋 당일 체결\n"
                f"  매수: {self.daily_buys}건 ({self.daily_buy_vol:.6f})\n"
                f"  매도: {self.daily_sells}건 ({self.daily_sell_vol:.6f})\n"
                f"{'─' * 28}\n"
                f"상태: {self.prev_state}"
            )

            self.notifier.send(msg)
            if pnl_available:
                self._log(f"📊 일일 리포트 발송 | 총 손익={total_pnl:+,.2f} USDT")
            else:
                self._log("📊 일일 리포트 발송 (PnL 조회 실패, 간소화 리포트)")
            self._report_sent_date = today

            # 리포트 발송 후 날짜가 바뀌었으면 카운터 리셋
            if is_new_day:
                self.daily_buys = 0
                self.daily_sells = 0
                self.daily_buy_vol = 0.0
                self.daily_sell_vol = 0.0
                self.daily_buy_cost = 0.0
                self.daily_sell_revenue = 0.0
                self.daily_fees = 0.0
                self.daily_realized = 0.0
        except Exception as e:
            self._log(f"일일 리포트 생성 실패: {e}", level="ERROR")

    # ─── 수수료 컨텍스트 ─────────────────────────────────────

    def _position_metrics(self, current_price: float) -> dict:
        """Return risk/PnL metrics using current inventory, not stale entry_price."""
        try:
            qty = max(float(self.holding_qty or 0.0), 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            holding_cost = max(float(self.holding_cost or 0.0), 0.0)
        except (TypeError, ValueError):
            holding_cost = 0.0
        try:
            realized = float(self.realized_pnl or 0.0)
        except (TypeError, ValueError):
            realized = 0.0
        try:
            entry = float(self.entry_price) if self.entry_price not in (None, "") else 0.0
        except (TypeError, ValueError):
            entry = 0.0

        avg_buy = holding_cost / qty if qty > 0 else 0.0
        exposure = qty * current_price if qty > 0 else 0.0
        unrealized = (current_price - avg_buy) * qty if avg_buy > 0 else 0.0
        total_pnl = realized + unrealized
        grid_budget = float(getattr(config, "GRID_BUDGET", 0.0) or 0.0)
        total_pnl_pct = (total_pnl / grid_budget * 100) if grid_budget > 0 else 0.0
        price_loss_pct = ((avg_buy - current_price) / avg_buy * 100) if avg_buy > 0 else 0.0
        entry_loss_pct = ((entry - current_price) / entry * 100) if entry > 0 else 0.0

        return {
            "qty": qty,
            "holding_cost": holding_cost,
            "avg_buy": avg_buy,
            "exposure": exposure,
            "unrealized": unrealized,
            "realized": realized,
            "total_pnl": total_pnl,
            "grid_budget": grid_budget,
            "total_pnl_pct": total_pnl_pct,
            "price_loss_pct": price_loss_pct,
            "entry_price": entry,
            "entry_loss_pct": entry_loss_pct,
        }

    def _stop_loss_status(self, current_price: float) -> dict:
        """Stop-loss status from average cost and total grid drawdown."""
        metrics = self._position_metrics(current_price)
        threshold = float(getattr(config, "MAX_LOSS_PERCENT", 15.0) or 15.0)
        reasons = []

        if metrics["avg_buy"] > 0 and metrics["price_loss_pct"] >= threshold:
            reasons.append(
                f"평균단가 대비 손실 {metrics['price_loss_pct']:.2f}% >= {threshold:.2f}%"
            )
        if metrics["grid_budget"] > 0 and metrics["total_pnl_pct"] <= -threshold:
            reasons.append(
                f"그리드 예산 대비 총손익 {metrics['total_pnl_pct']:.2f}% <= -{threshold:.2f}%"
            )
        if metrics["avg_buy"] <= 0 and metrics["entry_price"] > 0 and metrics["entry_loss_pct"] >= threshold:
            reasons.append(
                f"진입가 대비 손실 {metrics['entry_loss_pct']:.2f}% >= {threshold:.2f}%"
            )

        basis_price = metrics["avg_buy"] or metrics["entry_price"]
        basis = "avg_cost" if metrics["avg_buy"] > 0 else "entry_price"
        return {
            **metrics,
            "basis_price": basis_price,
            "basis": basis,
            "triggered": bool(reasons),
            "reason": "; ".join(reasons),
        }

    def _build_fee_context(self, current_price: float) -> str:
        """에이전트에게 제공할 수수료/손익/운영 컨텍스트.

        그리드봇 헌법 ④: LLM이 "정지의 비용"을 숫자로 인식하도록
        누적 수익, 일평균 수익, 일일 기회비용, 운영 기간을 함께 주입.
        """
        metrics = self._position_metrics(current_price)
        unrealized = metrics["unrealized"]
        avg_buy = metrics["avg_buy"]
        total_pnl = metrics["total_pnl"]
        total_pnl_pct = metrics["total_pnl_pct"]
        price_loss_pct = metrics["price_loss_pct"]

        # 1시간 내 그리드 재시작 횟수
        now = datetime.now()
        recent_restarts = [t for t in self.grid_restart_times
                           if (now - t).total_seconds() < 3600]

        # 예상 재시작 수수료 (보유분 매도 + 새 주문 체결 = 약 0.2%)
        est_restart_fee = self.holding_qty * current_price * 0.002 if self.holding_qty > 0 else 0

        # ─── 봇 운영 컨텍스트 (그리드봇 헌법 ④) ───
        # 운영 일수 추정: 봇 생성 시각이 있으면 사용, 없으면 누적 수수료 기반 보수적 추정
        running_days = self._estimate_running_days()
        avg_daily_pnl = (self.realized_pnl / running_days) if running_days > 0 else 0.0
        # 정지의 일일 기회비용 = 최근 일평균 실현 손익 (양수일 때만 의미)
        opportunity_cost_per_day = max(avg_daily_pnl, 0.0)
        # 당일 체결 횟수
        today_fills = self.daily_buys + self.daily_sells

        return (
            f"\n=== 봇 운영 현황 (정지의 기회비용 인식용) ===\n"
            f"누적 실현 손익: {self.realized_pnl:+,.2f} USDT\n"
            f"총 손익(실현+미실현): {total_pnl:+,.2f} USDT ({total_pnl_pct:+.2f}% of grid budget)\n"
            f"누적 수수료: {self.total_fees_paid:,.2f} USDT\n"
            f"운영 기간(추정): {running_days:.1f}일\n"
            f"일평균 실현 손익: {avg_daily_pnl:+,.2f} USDT/일\n"
            f"⚡ 정지 시 일일 기회비용: ~{opportunity_cost_per_day:,.2f} USDT/일\n"
            f"   (봇이 돌면 하루에 평균 이만큼 번다는 뜻 — STOP은 이만큼을 매일 포기)\n"
            f"\n=== 당일 활동 ===\n"
            f"당일 체결: 매수 {self.daily_buys}회 + 매도 {self.daily_sells}회 = {today_fills}건\n"
            f"당일 실현 손익: {self.daily_realized:+,.4f} USDT\n"
            f"당일 누적 수수료: {self.daily_fees:,.4f} USDT\n"
            f"당일 그리드 재시작: {self.grid_restart_count}회\n"
            f"최근 1시간 재시작: {len(recent_restarts)}회\n"
            f"\n=== 포지션 ===\n"
            f"미실현 손익: {unrealized:+,.4f} USDT\n"
            f"평균단 대비 가격 손익: {-price_loss_pct:+.2f}%\n"
            f"보유 수량: {self.holding_qty:.6f} (평균 매수가: {avg_buy:,.2f})\n"
            f"\n=== 액션 비용 ===\n"
            f"WIDEN/SHIFT 시 예상 수수료: ~{est_restart_fee:,.2f} USDT (1회)\n"
            f"STOP 시 비용: 위 수수료 + 일일 기회비용 영구 손실 (봇이 돌면 매일 벌었을 금액)\n"
            f"⚠ 그리드봇은 횡보로 돈 벌므로, 단발성 변동성 spike는 STOP 사유가 되지 않습니다."
        )

    def _estimate_running_days(self) -> float:
        """봇 운영 기간 추정 (일 단위).

        controller에 봇 생성 시각이 있으면 그걸 사용, 없으면 1.0 반환.
        """
        bot_ctime = getattr(self.controller, "bot_ctime", None)
        if isinstance(bot_ctime, datetime):
            elapsed = (datetime.now() - bot_ctime).total_seconds()
            return max(elapsed / 86400, 1.0)
        # 기본값: 1일 (분모 0 방지)
        return 1.0

    def _compose_bot_label(self) -> str:
        """봇 식별 라벨 자동 생성. 멀티봇 환경에서 텔레그램 알림 식별용.

        우선순위:
          1. config.BOT_LABEL (사용자 지정 별칭) — 설정되어 있으면 그대로 사용
          2. controller의 현재 그리드 범위 — 런타임 SHIFT/WIDEN 반영
          3. config.GRID_LOWER/UPPER — 봇 시작 전 fallback

        예: "ETH 2000-2400"
        """
        explicit = getattr(config, "BOT_LABEL", "") or ""
        if explicit:
            return explicit
        try:
            base = (config.SYMBOL or "").split("-")[0] or "BOT"
            # 런타임 그리드 범위 우선 (SHIFT/WIDEN으로 바뀐 값 반영)
            lo = getattr(self.controller, "current_lower", None)
            hi = getattr(self.controller, "current_upper", None)
            if lo is None or hi is None:
                lo = config.GRID_LOWER
                hi = config.GRID_UPPER
            return f"{base} {int(lo)}-{int(hi)}"
        except (ValueError, TypeError, AttributeError):
            return config.SYMBOL or "BOT"

    def _refresh_bot_label(self):
        """그리드 범위가 바뀌었을 수 있을 때 라벨 갱신.

        WIDEN/SHIFT 직후 또는 매 틱 시작 시 호출.
        Notifier에도 즉시 반영해서 다음 알림부터 새 라벨 prefix가 붙는다.
        """
        new_label = self._compose_bot_label()
        if new_label != self.bot_label:
            self._log(f"봇 라벨 갱신: {self.bot_label} → {new_label}")
            self.bot_label = new_label
            self.notifier.set_label(new_label)

    def _attach_bot_label(self, signal):
        """멀티 에이전트 컨텍스트에 봇 라벨 주입.

        multi_agent._build_market_context는 signal.bot_label을 읽어
        시장 데이터 헤더에 [봇: <label>]을 박는다 — LLM이 어느 봇 판단인지 인식.
        """
        try:
            setattr(signal, "bot_label", self.bot_label)
        except (AttributeError, TypeError):
            pass
        return signal

    def _get_bot_list_footer(self) -> str:
        """OKX의 활성 그리드봇 리스트를 텔레그램 footer 문자열로 반환.

        멀티봇 운영 시 매 알림에서 어떤 봇들이 돌고 있는지 보여준다.
        OKX API 호출이 빈번해지지 않도록 60초 캐시.
        실패 시 빈 문자열 반환 → footer 생략 (알림 자체 차단 안 함).
        """
        import time
        now = time.time()
        cached_at, cached_str = self._bot_list_cache
        if now - cached_at < 60 and cached_str:
            return cached_str

        try:
            bots = self.controller.list_active_bots()
        except (AttributeError, Exception):
            bots = []

        if not bots:
            return ""

        lines = ["─── 운영 중 봇 ───"]
        for b in bots:
            try:
                lo = float(b.get('minPx', 0))
                hi = float(b.get('maxPx', 0))
                state = b.get('state', '?')
                state_emoji = '🟢' if state == 'running' else '🔴' if state == 'stopped' else '⚪'
                grid_profit = float(b.get('gridProfit', 0) or 0)
                arb = b.get('arbitrageNum', '0')
                pnl_emoji = '📈' if grid_profit >= 0 else '📉'
                lines.append(
                    f"{state_emoji} {int(lo)}-{int(hi)} "
                    f"({arb} RT, {pnl_emoji} {grid_profit:+,.0f} USDT)"
                )
            except (ValueError, TypeError, KeyError):
                continue

        result = "\n".join(lines)
        self._bot_list_cache = (now, result)
        return result

    def _check_restart_allowed(self, action: str, current_price: float) -> tuple[bool, str]:
        """그리드 재시작이 수수료 대비 합리적인지 체크."""
        if action not in ("WIDEN", "SHIFT_UP", "SHIFT_DOWN"):
            return True, ""

        now = datetime.now()
        recent = [t for t in self.grid_restart_times
                  if (now - t).total_seconds() < 3600]

        # 하드 리밋: 1시간 내 최대 2회
        if len(recent) >= 2:
            return False, f"1시간 내 재시작 {len(recent)}회 도달 (최대 2회)"

        # 수수료 가드: 예상 수수료 > 최근 실현 수익이면 차단.
        # WIDEN is especially dangerous because it restarts the bot while keeping
        # the same inventory risk; require either realized edge or a breakout/risk reason.
        metrics = self._position_metrics(current_price)
        est_fee = metrics["exposure"] * 0.002 if metrics["exposure"] > 0 else 0.0
        if self.daily_realized > 0 and est_fee > self.daily_realized * 0.5:
            return False, (
                f"수수료 비효율: 예상 수수료 ~{est_fee:,.4f} > "
                f"실현 수익의 50% ({self.daily_realized * 0.5:,.4f})"
            )
        if action == "WIDEN" and self.daily_realized <= 0 and metrics["total_pnl"] < 0:
            return False, (
                f"WIDEN 보류: 당일 실현 수익이 없고 총손익이 "
                f"{metrics['total_pnl']:+,.2f} USDT"
            )

        return True, ""

    def _record_grid_restart(self):
        """그리드 재시작 기록."""
        self.grid_restart_times.append(datetime.now())
        self.grid_restart_count += 1

    # ─── 상태 저장 / 복원 ──────────────────────────────────

    @staticmethod
    def _state_path() -> Optional[str]:
        """설정된 상태 파일 경로. 빈 문자열이면 None."""
        path = (config.STATE_FILE or "").strip()
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        return path

    @staticmethod
    def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if isinstance(dt, datetime) else None

    @staticmethod
    def _str_to_dt(s) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    def _save_state(self):
        """틱 종료 시 현재 상태를 파일에 저장. 실패해도 루프는 계속."""
        path = self._state_path()
        if not path:
            return
        try:
            state = {
                "saved_at": datetime.now().isoformat(),
                "current_date": self._current_date,
                # 이탈 타이머
                "grid_breakout_time": self._dt_to_str(self.grid_breakout_time),
                "grid_breakout_dir": self.grid_breakout_dir,
                "grid_breakout_notified": self.grid_breakout_notified,
                # 재시작 추적
                "grid_restart_times": [
                    self._dt_to_str(t) for t in self.grid_restart_times
                ],
                "grid_restart_count": self.grid_restart_count,
                "total_fees_paid": self.total_fees_paid,
                "last_shift_time": self._dt_to_str(self.last_shift_time),
                # 당일 카운터
                "daily_buys": self.daily_buys,
                "daily_sells": self.daily_sells,
                "daily_buy_vol": self.daily_buy_vol,
                "daily_sell_vol": self.daily_sell_vol,
                "daily_buy_cost": self.daily_buy_cost,
                "daily_sell_revenue": self.daily_sell_revenue,
                "daily_fees": self.daily_fees,
                "daily_realized": self.daily_realized,
                "report_sent_date": self._report_sent_date,
                # 포지션 / 체결
                "holding_qty": self.holding_qty,
                "holding_cost": self.holding_cost,
                "realized_pnl": self.realized_pnl,
                "entry_price": self.entry_price,
                "last_fill_id": self.last_fill_id,
                # 메타
                "symbol": config.SYMBOL,
            }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            self._log(f"상태 저장 실패: {e}", level="ERROR")

    def _load_state(self):
        """시작 시 저장된 상태 복원. 날짜가 바뀌었으면 daily만 리셋."""
        path = self._state_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            self._log(f"상태 파일 읽기 실패: {e} → 초기 상태로 진행", level="ERROR")
            return

        # 심볼이 다르면 파일은 다른 봇 것. 무시.
        saved_symbol = state.get("symbol")
        if saved_symbol and saved_symbol != config.SYMBOL:
            self._log(
                f"저장된 심볼({saved_symbol})이 현재({config.SYMBOL})와 달라 상태 복원 건너뜀",
                level="WARNING"
            )
            return

        today = datetime.now().strftime("%Y-%m-%d")
        saved_date = state.get("current_date")
        same_day = (saved_date == today)

        # 항상 복원 (이탈 타이머 / 포지션 / 재시작 기록 / 누적 수수료)
        self.grid_breakout_time = self._str_to_dt(state.get("grid_breakout_time"))
        self.grid_breakout_dir = state.get("grid_breakout_dir")
        self.grid_breakout_notified = bool(state.get("grid_breakout_notified", False))
        self.grid_restart_times = [
            dt for dt in (self._str_to_dt(s) for s in state.get("grid_restart_times", []))
            if dt is not None
        ]
        self.total_fees_paid = float(state.get("total_fees_paid", 0.0))
        self.last_shift_time = self._str_to_dt(state.get("last_shift_time"))
        self.holding_qty = float(state.get("holding_qty", 0.0))
        self.holding_cost = float(state.get("holding_cost", 0.0))
        self.realized_pnl = float(state.get("realized_pnl", 0.0))
        raw_entry_price = state.get("entry_price")
        try:
            self.entry_price = (
                float(raw_entry_price)
                if raw_entry_price not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            self.entry_price = None
        self.last_fill_id = state.get("last_fill_id")

        # 당일 카운터: 같은 날짜일 때만 복원
        if same_day:
            self.daily_buys = int(state.get("daily_buys", 0))
            self.daily_sells = int(state.get("daily_sells", 0))
            self.daily_buy_vol = float(state.get("daily_buy_vol", 0.0))
            self.daily_sell_vol = float(state.get("daily_sell_vol", 0.0))
            self.daily_buy_cost = float(state.get("daily_buy_cost", 0.0))
            self.daily_sell_revenue = float(state.get("daily_sell_revenue", 0.0))
            self.daily_fees = float(state.get("daily_fees", 0.0))
            self.daily_realized = float(state.get("daily_realized", 0.0))
            self.grid_restart_count = int(state.get("grid_restart_count", 0))
            self._report_sent_date = state.get("report_sent_date")
            self._current_date = saved_date
        else:
            # 날짜 바뀐 경우 daily_*는 0부터 시작 (__init__ 기본값 유지)
            self._log(
                f"저장된 날짜({saved_date}) ≠ 오늘({today}) → 일일 카운터는 초기화",
                level="INFO"
            )

        breakout_info = ""
        if self.grid_breakout_time:
            elapsed_hr = (datetime.now() - self.grid_breakout_time).total_seconds() / 3600
            breakout_info = (
                f" | 이탈 타이머 복원: {self.grid_breakout_time.strftime('%m-%d %H:%M')} "
                f"({self.grid_breakout_dir}, {elapsed_hr:.1f}h 경과)"
            )
        self._log(
            f"✅ 상태 복원 완료 | 저장={state.get('saved_at', '?')}{breakout_info}"
        )

    # ─── 그리드 이탈 감지 & 대응 ─────────────────────────────

    def _check_grid_breakout(self, signal, price: float) -> Optional[str]:
        """
        가격이 그리드 범위를 이탈했는지 감지.
        - 이탈 직후: 알림 + 대기
        - BREAKOUT_WAIT_HOURS 이상 이탈: 에이전트에게 재배치 판단 요청
        - 복귀 시: 타이머 리셋
        Returns: 오버라이드할 액션 or None (정상 흐름)
        """
        gl = self.controller.current_lower
        gu = self.controller.current_upper
        if gl is None or gu is None:
            return None

        now = datetime.now()

        # ── 범위 안이면 이탈 상태 리셋 ──
        if gl <= price <= gu:
            if self.grid_breakout_time is not None:
                elapsed = (now - self.grid_breakout_time).total_seconds()
                self._log(f"✅ 그리드 범위 복귀 (이탈 {elapsed/60:.0f}분 만)")
                self.notifier.send(
                    f"✅ 그리드 범위 복귀 | {config.SYMBOL}\n"
                    f"{'─' * 28}\n"
                    f"현재가: {price:,.2f} USDT\n"
                    f"범위: {gl:,.2f} ~ {gu:,.2f}\n"
                    f"이탈 시간: {elapsed/60:.0f}분\n"
                    f"→ 자동 매매 재개"
                )
                self.grid_breakout_time = None
                self.grid_breakout_dir = None
                self.grid_breakout_notified = False
            return None

        # ── 이탈 감지 ──
        direction = "ABOVE" if price > gu else "BELOW"

        # 첫 이탈 감지
        if self.grid_breakout_time is None:
            self.grid_breakout_time = now
            self.grid_breakout_dir = direction
            self.grid_breakout_notified = False

        elapsed = (now - self.grid_breakout_time).total_seconds()
        elapsed_min = elapsed / 60
        elapsed_hr = elapsed / 3600
        wait_hr = self.BREAKOUT_WAIT_SEC / 3600

        # 이탈 알림 (최초 1회)
        if not self.grid_breakout_notified:
            self.grid_breakout_notified = True
            dir_emoji = "⬆️" if direction == "ABOVE" else "⬇️"
            dir_label = "상단 이탈" if direction == "ABOVE" else "하단 이탈"
            boundary = gu if direction == "ABOVE" else gl
            diff = abs(price - boundary)
            diff_pct = diff / boundary * 100

            self._log(f"⚠️ 그리드 {dir_label} | {price:,.2f} (경계: {boundary:,.2f})")
            self.notifier.send(
                f"{dir_emoji} 그리드 {dir_label} | {config.SYMBOL}\n"
                f"{'━' * 28}\n"
                f"현재가  : {price:,.2f} USDT\n"
                f"경계    : {boundary:,.2f} USDT\n"
                f"이탈 폭 : {diff:,.2f} ({diff_pct:.2f}%)\n"
                f"{'━' * 28}\n"
                f"범위: {gl:,.2f} ~ {gu:,.2f}\n"
                f"{'─' * 28}\n"
                f"⏳ {wait_hr:.0f}시간 대기 후 재배치 여부 판단\n"
                f"가격이 범위로 돌아오면 자동 매매 재개"
            )

        # ── 하드 타임아웃: LLM 무시하고 강제 재배치 ──
        if elapsed >= self.BREAKOUT_HARD_TIMEOUT_SEC:
            hard_hr = self.BREAKOUT_HARD_TIMEOUT_SEC / 3600
            self._log(
                f"🚨 이탈 {elapsed_hr:.1f}h ≥ 하드 타임아웃 {hard_hr:.0f}h "
                f"→ LLM 건너뛰고 강제 재배치",
                level="WARNING"
            )
            forced_action = "SHIFT_UP" if direction == "ABOVE" else "SHIFT_DOWN"
            allowed, skip_reason = self._check_restart_allowed(forced_action, price)
            if not allowed:
                self._log(f"수수료 가드 차단: {skip_reason} → 대기 유지", level="WARNING")
                self.notifier.send(
                    f"🛡️ 하드 타임아웃 강제 재배치 차단 | {config.SYMBOL}\n"
                    f"이탈: {elapsed_hr:.1f}h\n"
                    f"사유: {skip_reason}"
                )
                # 타이머 1시간 뒤로 리셋 (매 틱 재시도 방지)
                self.grid_breakout_time = now - timedelta(
                    seconds=self.BREAKOUT_HARD_TIMEOUT_SEC - 3600
                )
                return "MAINTAIN"

            self._record_grid_restart()
            old_lower, old_upper = gl, gu
            self.controller.shift_grid_center(price, price)
            self._refresh_bot_label()
            self.grid_breakout_time = None
            self.grid_breakout_dir = None
            self.grid_breakout_notified = False
            self.notifier.send(
                f"🚨 하드 타임아웃 강제 재배치 | {config.SYMBOL}\n"
                f"{'─' * 28}\n"
                f"이탈 {elapsed_hr:.1f}시간 경과 (한도 {hard_hr:.0f}h)\n"
                f"이전: {old_lower:,.2f} ~ {old_upper:,.2f}\n"
                f"새 범위: {self.controller.current_lower:,.2f} ~ "
                f"{self.controller.current_upper:,.2f}\n"
                f"중심: {price:,.2f} USDT\n"
                f"방향: {direction}"
            )
            return "MAINTAIN"

        # ── BREAKOUT_WAIT_HOURS 이상 이탈 → 에이전트에게 재배치 판단 요청 ──
        if elapsed >= self.BREAKOUT_WAIT_SEC:
            self._log(f"그리드 이탈 {elapsed_hr:.1f}시간 경과 → 에이전트 재배치 판단 요청")

            # 수수료 가드 체크
            candidate_action = "SHIFT_UP" if direction == "ABOVE" else "SHIFT_DOWN"
            allowed, skip_reason = self._check_restart_allowed(candidate_action, price)
            if not allowed:
                self._log(f"수수료 가드: {skip_reason} → 대기 유지")
                return "MAINTAIN"

            # 에이전트 판단
            fee_ctx = self._build_fee_context(price)
            breakout_ctx = (
                f"\n=== 그리드 이탈 상황 ===\n"
                f"이탈 방향: {'상단 (가격이 그리드 위)' if direction == 'ABOVE' else '하단 (가격이 그리드 아래)'}\n"
                f"이탈 시간: {elapsed_hr:.1f}시간\n"
                f"현재 그리드: {gl:,.2f} ~ {gu:,.2f}\n"
                f"현재가: {price:,.2f}\n"
                f"그리드 범위로 돌아올 가능성과 재배치 수수료를 비교해서 판단하세요.\n"
                f"재배치 = WIDEN/SHIFT, 계속 대기 = MAINTAIN\n"
            )

            try:
                if config.MULTI_AGENT_MODE and self.multi_agent.available:
                    result = self.multi_agent.judge_with_detail(
                        signal, price, fee_context=fee_ctx + breakout_ctx
                    )
                    action = result.final_action
                    self.notifier.send(
                        f"🤖 이탈 재배치 판단 | {config.SYMBOL}\n"
                        f"이탈: {elapsed_hr:.1f}시간 ({direction})\n"
                        f"에이전트 결정: {action}\n"
                        f"합의율: {result.agreement_rate:.0f}%\n"
                        f"사유: {result.reasoning}"
                    )
                else:
                    action = self.llm_judge.judge(
                        signal, price, fee_context=fee_ctx + breakout_ctx
                    )
                    self.notifier.send(
                        f"🤖 이탈 재배치 판단 | {config.SYMBOL}\n"
                        f"이탈: {elapsed_hr:.1f}시간 ({direction})\n"
                        f"LLM 결정: {action}"
                    )

                if action in ("WIDEN", "SHIFT_UP", "SHIFT_DOWN"):
                    # 재배치 실행. WIDEN keeps the action semantics; SHIFT recenters.
                    self._record_grid_restart()
                    old_lower, old_upper = gl, gu
                    if action == "WIDEN":
                        self.controller.widen_grid(signal.atr_current, price)
                    else:
                        self.controller.shift_grid_center(price, price)
                    self._refresh_bot_label()
                    self.grid_breakout_time = None
                    self.grid_breakout_dir = None
                    self.grid_breakout_notified = False
                    self.notifier.send(
                        f"🔄 이탈 후 그리드 재배치 | {config.SYMBOL}\n"
                        f"{'─' * 28}\n"
                        f"액션: {action}\n"
                        f"이전: {old_lower:,.2f} ~ {old_upper:,.2f}\n"
                        f"새 범위: {self.controller.current_lower:,.2f} ~ "
                        f"{self.controller.current_upper:,.2f}\n"
                        f"중심: {price:,.2f} USDT\n"
                        f"이탈 시간: {elapsed_hr:.1f}시간"
                    )
                    return "MAINTAIN"  # 재배치 완료, 정상 흐름
                else:
                    # 에이전트가 대기 결정 → 타이머를 1시간 뒤로 리셋 (매 틱 재판단 방지)
                    self.grid_breakout_time = now - timedelta(
                        seconds=self.BREAKOUT_WAIT_SEC - 3600
                    )
                    return "MAINTAIN"

            except Exception as e:
                self._log(f"이탈 재배치 판단 실패: {e}", level="ERROR")
                return "MAINTAIN"

        # 대기 시간 미만 이탈 → 대기
        if self.loop_count % 5 == 0:  # 5틱마다 대기 알림
            self.notifier.send(
                f"⏳ 그리드 이탈 대기 중 | {config.SYMBOL}\n"
                f"방향: {'상단' if direction == 'ABOVE' else '하단'}\n"
                f"경과: {elapsed_min:.0f}분 / {self.BREAKOUT_WAIT_SEC//60}분\n"
                f"현재가: {price:,.2f} USDT\n"
                f"범위: {gl:,.2f} ~ {gu:,.2f}"
            )
        return "MAINTAIN"

    # ─── 틱 리포트 텔레그램 발송 ─────────────────────────────

    def _send_tick_report(self, signal, price: float, action: str,
                          trend: str, trend_strength: float):
        """매 틱마다 텔레그램으로 요약 발송. EMERGENCY 시 반복 알림.

        config.NOTIFY_TICK_REPORTS=False면 매 틱 요약은 건너뛰고
        EMERGENCY 반복 알림만 발송 (사용자가 알림 폭주를 호소한 4/27 피드백).
        이벤트성 알림(LLM 합의/체결/이탈/상태변화/일일)은 다른 경로로 발송됨.
        """
        state_emoji = {"NORMAL": "🟢", "CAUTION": "🟡", "WARNING": "🟠", "EMERGENCY": "🔴"}
        emoji = state_emoji.get(signal.state, "⚪")

        # 빠른 종료: 매 틱 요약을 끈 경우, EMERGENCY 반복만 처리하고 종료
        if not config.NOTIFY_TICK_REPORTS and signal.state != "EMERGENCY":
            return
        pnl_str = ""
        try:
            pnl = self.controller.get_grid_pnl()
            if pnl:
                total = pnl.get("total_pnl", 0)
                pnl_emoji = "📈" if total >= 0 else "📉"
                pnl_str = f"\n{pnl_emoji} 손익: {total:+,.2f} USDT"
        except Exception:
            pass

        loss_str = ""
        if self.entry_price and price:
            loss_pct = (price - self.entry_price) / self.entry_price * 100
            loss_str = f"\n진입가 대비: {loss_pct:+.2f}%"

        # 체결 기반 손익 섹션
        fill_section = ""
        if self.daily_buys > 0 or self.daily_sells > 0:
            # 미실현 손익 (보유분 평가)
            unrealized = 0.0
            if self.holding_qty > 0 and price:
                avg_buy = self.holding_cost / self.holding_qty
                unrealized = (price - avg_buy) * self.holding_qty
            unrealized_emoji = "📈" if unrealized >= 0 else "📉"
            total_pnl = self.daily_realized + unrealized
            total_emoji = "💰" if total_pnl >= 0 else "💸"

            fill_section = (
                f"\n{'─' * 28}\n"
                f"📋 당일 체결\n"
                f"  매수: {self.daily_buys}건 / {self.daily_buy_cost:,.2f} USDT\n"
                f"  매도: {self.daily_sells}건 / {self.daily_sell_revenue:,.2f} USDT\n"
                f"  수수료: {self.daily_fees:,.4f} USDT\n"
                f"  보유: {self.holding_qty:.6f}\n"
                f"{'─' * 28}\n"
                f"  실현 손익: {self.daily_realized:+,.4f} USDT\n"
                f"  {unrealized_emoji} 미실현: {unrealized:+,.4f} USDT\n"
                f"  {total_emoji} 합계: {total_pnl:+,.4f} USDT"
            )

        # 그리드봇 상태 & 포지션
        grid_section = ""
        gl = self.controller.current_lower
        gu = self.controller.current_upper
        bot_id = self.controller.bot_id
        if gl is not None and gu is not None and gu > gl:
            grid_range = gu - gl
            position_pct = (price - gl) / grid_range * 100
            position_pct = max(0, min(100, position_pct))

            # 위치 바 (10칸)
            bar_pos = int(position_pct / 10)
            bar = "░" * bar_pos + "●" + "░" * (10 - bar_pos)

            # 범위 이탈 감지
            if price > gu:
                pos_label = "⚠️ 상단 이탈!"
            elif price < gl:
                pos_label = "⚠️ 하단 이탈!"
            elif position_pct >= 80:
                pos_label = "상단 근접"
            elif position_pct <= 20:
                pos_label = "하단 근접"
            else:
                pos_label = "범위 내"

            # 봇 상태
            bot_status = "✅ 가동 중" if bot_id else "❌ 봇 없음"

            # 포지션 요약
            avg_buy_str = ""
            if self.holding_qty > 0 and self.holding_cost > 0:
                avg_buy = self.holding_cost / self.holding_qty
                avg_buy_str = f" (평균 {avg_buy:,.2f})"

            # 그리드 개수 & 간격
            gn = self.controller.current_grid_num
            gm = self.controller.current_mode or "?"
            spacing = grid_range / gn if gn and gn > 0 else 0
            grid_info = f"{gn}칸 ({gm})" if gn else "?"
            spacing_str = f" | 간격: {spacing:,.2f}" if spacing > 0 else ""

            # OKX 실제 포지션 조회
            pos = self.controller.get_grid_positions()
            coin = config.SYMBOL.split("-")[0]  # ETH-USDT → ETH

            # OKX 계좌 전체 잔고
            balances = self.controller.get_account_balance()

            # 포지션 테이블 (계좌 잔고 기준)
            now_dt = datetime.now()
            ampm = "오후" if now_dt.hour >= 12 else "오전"
            hr = now_dt.hour % 12 or 12
            now_ts = f"{now_dt.month}/{now_dt.day} {ampm} {hr}:{now_dt.minute:02d}"
            portfolio_lines = ""
            if balances:
                total_eq = 0.0
                rows = []
                for ccy, bal in sorted(balances.items()):
                    total_bal = bal.get("total", 0)
                    eq_usd = bal.get("eq_usd", 0)
                    if total_bal <= 0 and eq_usd <= 0:
                        continue
                    total_eq += eq_usd
                    if ccy == coin:
                        rows.append(
                            f"  {ccy:<8}{total_bal:>10.2f}개  {price:>10,.0f}  ~{eq_usd:>10,.0f} USDT"
                        )
                    elif ccy == "USDT":
                        rows.append(
                            f"  {ccy:<8}{total_bal:>10,.0f}     {'-':>10}  {total_bal:>11,.0f} USDT"
                        )
                    else:
                        if eq_usd >= 1:
                            rows.append(
                                f"  {ccy:<8}{total_bal:>10.4f}     {'-':>10}  ~{eq_usd:>10,.0f} USDT"
                            )

                # 그리드봇 손익 (OKX API 기준)
                bot_pnl_lines = ""
                if pos:
                    investment = pos.get("investment", 0)
                    grid_profit = pos.get("grid_profit", 0)
                    float_profit = pos.get("float_profit", 0)
                    total_pnl_bot = pos.get("total_pnl", 0)
                    pnl_pct = (total_pnl_bot / investment * 100) if investment > 0 else 0
                    pnl_emoji = "✅" if total_pnl_bot >= 0 else "🔻"
                    bot_pnl_lines = (
                        f"\n"
                        f"  그리드봇 투자금: {investment:,.0f} USDT\n"
                        f"  그리드 수익: {grid_profit:+,.2f} USDT\n"
                        f"  평가 손익:   {float_profit:+,.2f} USDT\n"
                        f"  봇 총손익:   {total_pnl_bot:+,.2f} USDT ({pnl_pct:+.2f}%) {pnl_emoji}"
                    )

                portfolio_lines = (
                    f"\n{'─' * 28}\n"
                    f"현재 포지션 ({now_ts})\n"
                    f"  {'통화':<8}{'보유량':>10}  {'현재가':>10}  {'평가액':>14}\n"
                    + "\n".join(rows)
                    + f"\n  {'총 평가':<8}{'':>10}  {'':>10}  ~{total_eq:>10,.0f} USDT"
                    + bot_pnl_lines
                )

            # 당일 체결 집계 (OKX API)
            fill_info = ""
            try:
                tf = self.controller.get_today_fills()
                bc = tf["buy_count"]
                sc = tf["sell_count"]
                rt = tf["round_trips"]
                net = tf["net_profit"]
                fees = tf["total_fees"]
                gross = tf.get("gross_per_trip", 0)
                fee_rt = tf.get("fee_per_trip", 0)
                net_rt = tf.get("net_per_trip", 0)
                net_emoji = "🔥" if net > 0 else "🧊" if net < 0 else ""

                fill_info = (
                    f"\n{'─' * 28}\n"
                    f"📊 오늘 체결\n"
                    f"  {'구분':<12}{'건수':>8}\n"
                    f"  {'매수':<12}{bc:>7}건\n"
                    f"  {'매도':<12}{sc:>7}건\n"
                    f"  {'왕복':<12}{rt:>7}회\n"
                    f"  {'수수료':<12}~{fees:,.2f} USDT\n"
                    f"  {'─' * 26}\n"
                    f"  1회 차익: ~{gross:,.2f} USDT\n"
                    f"  1회 수수료: ~{fee_rt:,.2f} USDT\n"
                    f"  1회 순수익: ~{net_rt:,.2f} USDT\n"
                    f"  {'─' * 26}\n"
                    f"  오늘 순수익: ~{net:+,.2f} USDT {net_emoji}"
                )
            except Exception:
                pass

            # 미체결 주문 그리드 시각화
            grid_visual = ""
            try:
                pending = self.controller.get_pending_orders()
                sells = pending.get("sell", [])
                buys = pending.get("buy", [])

                if sells or buys:
                    coin = config.SYMBOL.split("-")[0]
                    lines = []
                    lines.append(f"\n{'─' * 28}")
                    lines.append(f"{coin} 봇 현재 포지션 (현재가: {price:,.0f})")
                    lines.append("")

                    # 매도 주문 (높은 가격부터)
                    for s in sells:
                        px_int = int(s['price'])
                        diff = abs(px_int - int(price))
                        if diff <= 1:
                            lines.append(
                                f"  {px_int:,} — 매도 대기 ⬆ ← 현재가 {int(price):,} (거의 체결!!)"
                            )
                        else:
                            lines.append(f"  {px_int:,} — 매도 대기 ⬆")

                    # 현재가 라인
                    lines.append(f"  ----- 현재가 {int(price):,} -----")

                    # 매수 주문 (높은 가격부터)
                    for b in buys:
                        px_int = int(b['price'])
                        diff = abs(int(price) - px_int)
                        if diff <= 1:
                            lines.append(
                                f"  {px_int:,} — 매수 대기 ⬇ ← 현재가 {int(price):,} (거의 체결!!)"
                            )
                        else:
                            lines.append(f"  {px_int:,} — 매수 대기 ⬇")

                    # 포지션 요약
                    total_sell_sz = sum(s['size'] for s in sells)
                    total_sell_amt = sum(s['amount'] for s in sells)
                    total_buy_sz = sum(b['size'] for b in buys)
                    total_buy_amt = sum(b['amount'] for b in buys)

                    sell_prices = [int(s['price']) for s in sells]
                    buy_prices = [int(b['price']) for b in buys]
                    sell_range = f"{min(sell_prices):,}~{max(sell_prices):,}" if sell_prices else "-"
                    buy_range = f"{min(buy_prices):,}~{max(buy_prices):,}" if buy_prices else "-"

                    lines.append("")
                    lines.append("포지션 요약")
                    lines.append(f"  {'구분':<8}{'가격':<18}{'수량':<12}{'금액'}")
                    lines.append(
                        f"  {'매도대기':<8}{sell_range:<18}"
                        f"{total_sell_sz:.4f} {coin:<6}~{total_sell_amt:,.0f} USDT"
                    )
                    lines.append(
                        f"  {'매수대기':<8}{buy_range} ({len(buys)}칸)  "
                        f"{total_buy_sz:.4f} {coin:<6}~{total_buy_amt:,.0f} USDT"
                    )

                    # 가장 가까운 체결 알림
                    nearest_sell = sells[-1]['price'] if sells else None
                    nearest_buy = buys[0]['price'] if buys else None
                    if nearest_sell and abs(nearest_sell - price) <= 2:
                        lines.append(
                            f"\n🔥 {int(nearest_sell):,} 매도가 딱 "
                            f"{abs(nearest_sell - price):.0f} USDT 차이! "
                            f"조금만 올라오면 바로 체결!"
                        )
                    elif nearest_buy and abs(price - nearest_buy) <= 2:
                        lines.append(
                            f"\n🔥 {int(nearest_buy):,} 매수가 딱 "
                            f"{abs(price - nearest_buy):.0f} USDT 차이! "
                            f"조금만 내려오면 바로 체결!"
                        )

                    grid_visual = "\n".join(lines)
            except Exception:
                pass

            grid_section = (
                f"\n{'─' * 28}\n"
                f"🤖 그리드봇: {bot_status}\n"
                f"📐 {gl:,.2f} [{bar}] {gu:,.2f}\n"
                f"   위치: {position_pct:.0f}% ({pos_label})\n"
                f"   {grid_info}{spacing_str}"
                f"{fill_info}"
                f"{portfolio_lines}\n"
                f"🔄 재시작: 당일 {self.grid_restart_count}회 | 수수료: {self.daily_fees:,.4f}"
            )

        # 메시지 1: 틱 요약
        msg = (
            f"{emoji} TICK #{self.loop_count} | {config.SYMBOL}\n"
            f"{'━' * 28}\n"
            f"💰 {config.SYMBOL} : {price:,.2f} USDT\n"
            f"{'━' * 28}\n"
            f"상태: {signal.state} | 점수: {signal.risk_score:.1f}/100\n"
            f"추세: {trend} (ADX={trend_strength:.1f})\n"
            f"액션: {action}"
            f"{pnl_str}"
            f"{loss_str}"
            f"{grid_section}"
            f"{fill_section}\n"
            f"{'─' * 28}\n"
            f"ATR={signal.atr_current:.1f} | RSI={signal.rsi:.1f} | "
            f"BB={signal.bb_width:.2f}% | Vol={signal.volume_ratio:.1f}x"
        )
        self.notifier.send(msg)

        # 메시지 2: 그리드 주문 래더 (별도 메시지)
        if grid_visual:
            self.notifier.send(grid_visual)

        # EMERGENCY: 10초 간격으로 3회 반복 알림
        if signal.state == "EMERGENCY":
            emergency_msg = (
                f"🚨🚨🚨 긴급 알림 🚨🚨🚨\n\n"
                f"리스크 점수: {signal.risk_score:.1f}/100\n"
                f"현재가: {price:,.2f} USDT\n"
                f"사유: {signal.reason}\n"
                f"액션: {action}\n\n"
                f"즉시 확인이 필요합니다!"
            )
            for i in range(3):
                time.sleep(10)
                self.notifier.send(f"[{i+2}/4] {emergency_msg}")

    # ─── 대기 프로그레스 바 ────────────────────────────────────

    def _wait_with_progress(self, seconds: int):
        """다음 틱까지 프로그레스 바로 대기 시간 시각화."""
        BAR_WIDTH = 40
        DIM = "\033[2m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        for elapsed in range(seconds):
            remaining = seconds - elapsed
            progress = elapsed / seconds
            filled = int(BAR_WIDTH * progress)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)

            mins, secs = divmod(remaining, 60)
            time_str = f"{mins}:{secs:02d}" if mins else f"{secs}초"

            print(
                f"\r  {DIM}⏳{RESET} {CYAN}{bar}{RESET} "
                f"{GREEN}{progress*100:5.1f}%{RESET} "
                f"{DIM}(다음 틱까지 {time_str}){RESET}",
                end="", flush=True
            )
            time.sleep(1)

        # 완료
        bar = "█" * BAR_WIDTH
        print(
            f"\r  {DIM}✓{RESET}  {GREEN}{bar}{RESET} "
            f"{GREEN}{BOLD}100.0%{RESET} "
            f"{DIM}(시작!){RESET}              "
        )

    # ─── 손절 체크 ─────────────────────────────────────────

    def _check_stop_loss(self, current_price: float) -> bool:
        """Return True when average-cost or total-grid drawdown reaches the stop limit."""
        if self.entry_price is None and self.holding_qty <= 0:
            self.entry_price = current_price
            return False
        return bool(self._stop_loss_status(current_price)["triggered"])

    # ─── 로그 ──────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from menu import main_menu, clear

    try:
        result = main_menu()
    except KeyboardInterrupt:
        print("\n사용자 중단 — 종료합니다.")
        sys.exit(0)

    if result == "start":
        clear()
        # 메뉴에서 설정 변경했을 수 있으므로 config 모듈 다시 로드
        import importlib
        import config
        importlib.reload(config)
        GridAgent().run()
