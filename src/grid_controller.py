"""
grid_controller.py
OKX REST API를 이용해 그리드봇을 제어합니다.
"""

import hmac, hashlib, base64, time, json
from datetime import datetime, timezone
from typing import Optional
import httpx

_MAX_RETRIES = 3
_RETRY_DELAY = 2

from config import (
    OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE,
    OKX_BASE_URL, DEMO_MODE, OKX_TIMEOUT_SEC,
    SYMBOL, GRID_LOWER, GRID_UPPER, GRID_COUNT, GRID_MODE,
    GRID_BUDGET, ATR_PERIOD
)


class GridController:
    """
    OKX Spot Grid Bot 제어 클래스.

    이벤트 기반 액션:
      MAINTAIN  → ensure_grid_running()
      WIDEN     → widen_grid()
      SHIFT     → shift_grid_center()
      STOP      → emergency_stop()
    """

    def __init__(self):
        self.bot_id: Optional[str] = None      # 실행 중인 봇 ID
        self.current_lower: Optional[float] = None   # 현재 그리드 하한
        self.current_upper: Optional[float] = None   # 현재 그리드 상한
        self.current_grid_num: Optional[int] = None  # 현재 그리드 개수
        self.current_mode: Optional[str] = None      # arithmetic / geometric
        self.client = httpx.Client(
            base_url=OKX_BASE_URL,
            timeout=httpx.Timeout(OKX_TIMEOUT_SEC, connect=10.0),
        )

    # ─── 유틸리티 ─────────────────────────────────────────────

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Safely parse a float value from an API response."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # ─── 기존 봇 동기화 ────────────────────────────────────────

    def sync_existing_bot(self) -> dict:
        """
        OKX에서 현재 심볼에 대해 실행 중인 그리드봇을 조회하고,
        있으면 에이전트 상태를 동기화합니다.
        Returns: 동기화 결과 dict (status, bot_id 등)
        """
        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/orders-algo-pending",
                params={"algoOrdType": "grid"}
            )

            if resp.get("code") != "0":
                self._log(f"기존 봇 조회 실패: code={resp.get('code')} msg={resp.get('msg', '')}", level="ERROR")
                return {"status": "query_failed", "resp": resp}

            bots = resp.get("data", [])
            if not isinstance(bots, list):
                return {"status": "no_bots"}

            # 현재 심볼과 일치하는 봇 찾기
            for bot in bots:
                if not isinstance(bot, dict):
                    continue
                if bot.get("instId") != SYMBOL:
                    continue

                # 동기화
                self.bot_id = bot.get("algoId")
                self.current_lower = self._safe_float(bot.get("minPx"))
                self.current_upper = self._safe_float(bot.get("maxPx"))
                grid_num = bot.get("gridNum", "?")
                run_type = bot.get("runType", "1")
                mode = "arithmetic" if run_type == "1" else "geometric"
                try:
                    self.current_grid_num = int(grid_num)
                except (ValueError, TypeError):
                    self.current_grid_num = None
                self.current_mode = mode
                state = bot.get("state", "unknown")
                investment = self._safe_float(bot.get("investment"))
                total_pnl = self._safe_float(bot.get("totalPnl"))
                grid_profit = self._safe_float(bot.get("gridProfit"))
                float_pnl = self._safe_float(bot.get("floatProfit"))

                self._log(
                    f"✅ 기존 그리드봇 감지 | bot_id={self.bot_id}\n"
                    f"     심볼: {SYMBOL} | 상태: {state}\n"
                    f"     범위: {self.current_lower:,.2f} ~ {self.current_upper:,.2f}\n"
                    f"     그리드: {grid_num}개 ({mode})\n"
                    f"     투자금: {investment:,.2f} USDT\n"
                    f"     손익: 그리드={grid_profit:+,.2f} 평가={float_pnl:+,.2f} 합계={total_pnl:+,.2f}"
                )

                return {
                    "code": "0",
                    "status": "synced",
                    "bot_id": self.bot_id,
                    "lower": self.current_lower,
                    "upper": self.current_upper,
                    "grid_num": grid_num,
                    "mode": mode,
                    "state": state,
                    "investment": investment,
                    "total_pnl": total_pnl,
                }

            return {"status": "no_bots"}

        except Exception as e:
            self._log(f"기존 봇 동기화 실패: {e}", level="ERROR")
            return {"status": "error", "msg": str(e)}

    def list_active_bots(self) -> list[dict]:
        """OKX의 모든 활성 그리드봇 리스트 반환 (심볼 무관, 멀티봇 표시용).

        텔레그램 알림 footer에 어떤 봇들이 돌고 있는지 보여주는 용도.
        실패 시 빈 리스트 반환 (호출자가 footer 생략 처리).
        """
        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/orders-algo-pending",
                params={"algoOrdType": "grid"}
            )
            if resp.get("code") != "0":
                return []
            bots = resp.get("data", [])
            if not isinstance(bots, list):
                return []
            return [b for b in bots if isinstance(b, dict)]
        except Exception:
            return []

    # ─── 공개 액션 메서드 ────────────────────────────────────

    def ensure_grid_running(self, lower=None, upper=None, count=None) -> dict:
        """기존 봇 동기화 시도 → 없으면 새로 시작."""
        if self.bot_id:
            return {"code": "0", "status": "already_running", "bot_id": self.bot_id}

        # 먼저 OKX에서 기존 봇 확인
        sync = self.sync_existing_bot()
        if sync.get("status") == "synced":
            return sync
        if sync.get("status") not in ("no_bots",):
            self._log(
                f"기존 봇 조회가 실패하여 새 그리드 생성을 중단합니다 "
                f"(status={sync.get('status')}, msg={sync.get('msg', sync.get('resp', ''))})",
                level="ERROR",
            )
            return {
                "code": "-1",
                "status": "sync_failed",
                "msg": "existing grid lookup failed; refusing to start a duplicate bot",
                "sync": sync,
            }

        # 없으면 새로 시작
        return self.start_grid(lower, upper, count)

    def start_grid(self, lower=None, upper=None, count=None, mode=None) -> dict:
        """새 그리드봇을 시작합니다."""
        lower = lower or GRID_LOWER
        upper = upper or GRID_UPPER
        count = count or GRID_COUNT
        mode = mode or self.current_mode or GRID_MODE

        body = {
            "instId":       SYMBOL,
            "algoOrdType":  "grid",
            "maxPx":        str(upper),
            "minPx":        str(lower),
            "gridNum":      str(count),
            "runType":      "1" if mode == "arithmetic" else "2",
            "quoteSz":      str(GRID_BUDGET),
        }
        resp = self._post("/api/v5/tradingBot/grid/order-algo", body)

        if resp.get("code") == "0":
            try:
                data = resp.get("data")
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    self.bot_id = data[0].get("algoId")
                else:
                    self._log(f"그리드봇 시작 응답 구조 이상: {resp}", level="ERROR")
                self.current_lower = float(lower)
                self.current_upper = float(upper)
                self.current_grid_num = int(count)
                self.current_mode = mode
                self._log(f"그리드봇 시작 | bot_id={self.bot_id} | 범위={lower}~{upper} | {count}개 그리드")
            except Exception as e:
                self._log(f"그리드봇 시작 응답 파싱 실패: {e}", level="ERROR")
        else:
            sMsg = ""
            if isinstance(resp.get("data"), list) and resp["data"]:
                sMsg = resp["data"][0].get("sMsg", "")
            self._log(f"그리드봇 시작 실패: code={resp.get('code')} sMsg={sMsg}", level="ERROR")

        return resp

    def widen_grid(self, atr_value: float, current_price: float) -> dict:
        """
        CAUTION 상태: 그리드 간격을 ATR x 2배 기준으로 넓힙니다.
        기존 봇을 중지하고 더 넓은 범위로 재시작합니다.
        """
        if not self.bot_id:
            return {"status": "no_bot"}

        current_range = (
            self.current_upper - self.current_lower
            if self.current_lower is not None and self.current_upper is not None
            else GRID_UPPER - GRID_LOWER
        )
        # WIDEN must never shrink the active grid. ATR from 1m candles can be tiny,
        # so use it only as a floor alongside the current range.
        new_range = max(current_range * 1.25, atr_value * 8, current_price * 0.08)
        half_range = new_range / 2
        new_lower  = max(current_price - half_range, current_price * 0.2, 0.0001)
        new_upper  = current_price + half_range
        if new_upper <= new_lower:
            new_lower = current_price - current_range / 2
            new_upper = current_price + current_range / 2

        self._log(f"그리드 간격 확대 | 새 범위={new_lower:.0f}~{new_upper:.0f} (ATR={atr_value:.1f})")

        self.stop_grid(sell_remaining=False)
        return self.start_grid(
            lower=new_lower,
            upper=new_upper,
            count=self.current_grid_num or GRID_COUNT,
            mode=self.current_mode or GRID_MODE,
        )

    def emergency_stop(self) -> dict:
        """
        EMERGENCY 상태: 모든 포지션을 시장가로 즉시 청산합니다.
        """
        self._log("긴급 청산 실행 (EMERGENCY)", level="CRITICAL")
        result = self.stop_grid(sell_remaining=True)
        self.bot_id = None
        return result

    def stop_grid(self, sell_remaining: bool = False) -> dict:
        """그리드봇을 중지합니다."""
        if not self.bot_id:
            return {"status": "no_bot"}

        # OKX v5 stop-order-algo는 array of objects를 받음.
        # stopType: "1" = 기존 포지션 유지, "2" = 시장가 청산
        body = [{
            "algoId":      self.bot_id,
            "instId":      SYMBOL,
            "algoOrdType": "grid",
            "stopType":    "2" if sell_remaining else "1",
        }]

        resp = self._post("/api/v5/tradingBot/grid/stop-order-algo", body)

        if resp.get("code") == "0":
            self._log(f"그리드봇 중지 | sell_remaining={sell_remaining}")
            self.bot_id = None
        else:
            self._log(f"그리드봇 중지 실패: code={resp.get('code')} msg={resp.get('msg', '')}", level="ERROR")

        return resp

    def get_bot_status(self) -> dict:
        """현재 봇 상태와 PnL 조회."""
        if not self.bot_id:
            return {"status": "no_bot"}

        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/orders-algo-details",
                params={"algoId": self.bot_id, "algoOrdType": "grid"}
            )
            if not isinstance(resp, dict):
                return {"code": "-1", "msg": "unexpected response type"}
            return resp
        except Exception as e:
            self._log(f"get_bot_status 실패: {e}", level="ERROR")
            return {"code": "-1", "msg": str(e)}

    def get_recent_fills(self, limit: int = 20) -> list[dict]:
        """최근 체결 내역 조회."""
        try:
            resp = self._get(
                "/api/v5/trade/fills-history",
                params={"instId": SYMBOL, "limit": str(limit)}
            )
            data = resp.get("data", [])
            if not isinstance(data, list):
                self._log(f"get_recent_fills 응답 'data' 타입 이상: {type(data)}", level="ERROR")
                return []
            return data
        except Exception as e:
            self._log(f"get_recent_fills 실패: {e}", level="ERROR")
            return []

    def get_grid_pnl(self) -> dict:
        """그리드봇 수익 정보 조회."""
        if not self.bot_id:
            return {}
        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/orders-algo-details",
                params={"algoId": self.bot_id, "algoOrdType": "grid"}
            )
            if resp.get("code") == "0" and resp.get("data"):
                data_list = resp.get("data")
                if not isinstance(data_list, list) or len(data_list) == 0:
                    return {}
                data = data_list[0]
                if not isinstance(data, dict):
                    return {}
                return {
                    "grid_profit": self._safe_float(data.get("gridProfit")),
                    "float_profit": self._safe_float(data.get("floatProfit")),
                    "total_pnl": self._safe_float(data.get("totalPnl")),
                    "annualized_rate": self._safe_float(data.get("annualizedRate")),
                    "investment": self._safe_float(data.get("investment")),
                }
        except Exception as e:
            self._log(f"get_grid_pnl 실패: {e}", level="ERROR")
        return {}

    def get_account_balance(self) -> dict:
        """OKX 계좌 잔고 조회 (현물). 코인별 보유량 + USDT 잔고."""
        try:
            resp = self._get("/api/v5/account/balance")
            if resp.get("code") != "0" or not resp.get("data"):
                return {}

            data = resp["data"][0] if isinstance(resp["data"], list) and resp["data"] else {}
            details = data.get("details", [])
            if not isinstance(details, list):
                return {}

            balances = {}
            for d in details:
                if not isinstance(d, dict):
                    continue
                ccy = d.get("ccy", "")
                avail = self._safe_float(d.get("availBal"))
                frozen = self._safe_float(d.get("frozenBal"))
                total = self._safe_float(d.get("cashBal"))
                eq_usd = self._safe_float(d.get("eqUsd"))
                if total > 0 or avail > 0 or frozen > 0:
                    balances[ccy] = {
                        "available": avail,
                        "frozen": frozen,
                        "total": total,
                        "eq_usd": eq_usd,
                    }
            return balances
        except Exception as e:
            self._log(f"계좌 잔고 조회 실패: {e}", level="ERROR")
            return {}

    def get_grid_positions(self) -> dict:
        """그리드봇의 상세 포지션 정보 조회."""
        if not self.bot_id:
            return {}
        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/orders-algo-details",
                params={"algoId": self.bot_id, "algoOrdType": "grid"}
            )
            if resp.get("code") != "0" or not resp.get("data"):
                return {}
            data = resp["data"][0] if isinstance(resp["data"], list) else {}
            if not isinstance(data, dict):
                return {}

            return {
                "state": data.get("state", "unknown"),
                "investment": self._safe_float(data.get("investment")),
                "grid_profit": self._safe_float(data.get("gridProfit")),
                "float_profit": self._safe_float(data.get("floatProfit")),
                "total_pnl": self._safe_float(data.get("totalPnl")),
                "filled_count": data.get("filledCount", "0"),
                "total_count": data.get("totalCount", "0"),
                "annualized_rate": self._safe_float(data.get("annualizedRate")),
                "base_sz": self._safe_float(data.get("baseSz")),  # 보유 코인 수량
                "quote_sz": self._safe_float(data.get("quoteSz")),  # 투입 USDT
                "cur_base_sz": self._safe_float(data.get("curBaseSz")),  # 현재 코인 보유
                "cur_quote_sz": self._safe_float(data.get("curQuoteSz")),  # 현재 USDT 잔여
            }
        except Exception as e:
            self._log(f"그리드 포지션 조회 실패: {e}", level="ERROR")
            return {}

    def get_pending_orders(self) -> dict:
        """그리드봇 서브 주문을 매수/매도로 분류해서 반환."""
        if not self.bot_id:
            return {"buy": [], "sell": []}
        try:
            resp = self._get(
                "/api/v5/tradingBot/grid/sub-orders",
                params={
                    "algoId": self.bot_id,
                    "algoOrdType": "grid",
                    "type": "live",
                }
            )
            orders = resp.get("data", [])
            if not isinstance(orders, list):
                return {"buy": [], "sell": []}

            buys = []
            sells = []
            for o in orders:
                if not isinstance(o, dict):
                    continue
                side = o.get("side", "")
                px = self._safe_float(o.get("px"))
                sz = self._safe_float(o.get("sz"))
                if px <= 0:
                    continue
                entry = {"price": px, "size": sz, "amount": px * sz}
                if side == "buy":
                    buys.append(entry)
                elif side == "sell":
                    sells.append(entry)

            buys.sort(key=lambda x: x["price"], reverse=True)
            sells.sort(key=lambda x: x["price"], reverse=True)
            return {"buy": buys, "sell": sells}
        except Exception as e:
            self._log(f"그리드봇 서브 주문 조회 실패: {e}", level="ERROR")
            return {"buy": [], "sell": []}

    def get_today_fills(self) -> dict:
        """당일 그리드봇 체결 내역 집계.
        왕복(round trip) = min(매수, 매도), 순수익 = 왕복 × (간격 × 수량) - 수수료.
        """
        empty = {"buy_count": 0, "sell_count": 0, "total_count": 0,
                 "round_trips": 0, "gross_per_trip": 0, "fee_per_trip": 0,
                 "net_per_trip": 0, "net_profit": 0, "total_fees": 0}
        if not self.bot_id:
            return empty
        try:
            # 당일 00:00 기준 밀리초
            now = datetime.now()
            today_start = datetime(now.year, now.month, now.day)
            today_ms = int(today_start.timestamp() * 1000)

            # 페이지네이션으로 당일 체결 전부 수집 (최대 5페이지 = 500건)
            orders = []
            after = ""
            for _ in range(5):
                params = {
                    "algoId": self.bot_id,
                    "algoOrdType": "grid",
                    "type": "filled",
                }
                if after:
                    params["after"] = after
                resp = self._get("/api/v5/tradingBot/grid/sub-orders", params)
                page = resp.get("data", [])
                if not isinstance(page, list) or not page:
                    break
                # 당일 이전 데이터 나오면 중단
                oldest_time = int(page[-1].get("fillTime", page[-1].get("uTime", 0)))
                orders.extend(page)
                if oldest_time < today_ms:
                    break
                # 다음 페이지 커서
                after = page[-1].get("ordId", "")
                if not after:
                    break

            buy_count = 0
            sell_count = 0
            buy_fees_coin = 0.0   # 매수 수수료 (코인 단위)
            sell_fees_usdt = 0.0  # 매도 수수료 (USDT 단위)
            sizes = []
            last_price = 0.0

            for o in orders:
                if not isinstance(o, dict):
                    continue
                fill_time = int(o.get("fillTime", o.get("uTime", 0)))
                if fill_time < today_ms:
                    continue

                side = o.get("side", "")
                px = self._safe_float(o.get("px"))
                sz = self._safe_float(o.get("sz"))
                fee = abs(self._safe_float(o.get("fee")))
                sizes.append(sz)
                if px > 0:
                    last_price = px

                if side == "buy":
                    buy_count += 1
                    buy_fees_coin += fee  # ETH로 차감
                elif side == "sell":
                    sell_count += 1
                    sell_fees_usdt += fee  # USDT로 차감

            # 매수 수수료를 현재가로 USDT 환산
            buy_fees_usdt = buy_fees_coin * last_price if last_price > 0 else 0
            total_fees = buy_fees_usdt + sell_fees_usdt

            # 왕복 = 완성된 매수→매도 사이클
            round_trips = min(buy_count, sell_count)

            # 그리드 간격
            spacing = 0.0
            if (self.current_lower is not None and self.current_upper is not None
                    and self.current_grid_num and self.current_grid_num > 0):
                spacing = (self.current_upper - self.current_lower) / self.current_grid_num

            # 평균 체결 수량
            avg_sz = sum(sizes) / len(sizes) if sizes else 0

            # 1회 왕복 수익
            gross_per_trip = spacing * avg_sz
            fee_per_trip = total_fees / round_trips if round_trips > 0 else 0
            net_per_trip = gross_per_trip - fee_per_trip

            # 총 순수익
            net_profit = round_trips * net_per_trip

            return {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "total_count": buy_count + sell_count,
                "round_trips": round_trips,
                "avg_size": avg_sz,
                "spacing": spacing,
                "gross_per_trip": gross_per_trip,
                "fee_per_trip": fee_per_trip,
                "net_per_trip": net_per_trip,
                "net_profit": net_profit,
                "total_fees": total_fees,
            }
        except Exception as e:
            self._log(f"당일 체결 집계 실패: {e}", level="ERROR")
            return empty

    # ─── 그리드 중심 이동 & 노출 축소 ──────────────────────────

    def shift_grid_center(self, new_center: float, current_price: float,
                          grid_range: float = None) -> dict:
        """
        그리드 중심을 new_center로 이동합니다 (trailing grid).
        grid_range가 None이면 현재 GRID_UPPER - GRID_LOWER 폭을 그대로 사용합니다.
        """
        if not self.bot_id:
            return {"status": "no_bot"}

        if grid_range is None:
            if self.current_lower is not None and self.current_upper is not None:
                grid_range = self.current_upper - self.current_lower
            else:
                grid_range = GRID_UPPER - GRID_LOWER

        new_lower = new_center - grid_range / 2
        new_upper = new_center + grid_range / 2

        self._log(
            f"그리드 중심 이동 | new_center={new_center:.2f} "
            f"| 새 범위={new_lower:.2f}~{new_upper:.2f} "
            f"| current_price={current_price:.2f}"
        )

        self.stop_grid(sell_remaining=False)
        resp = self.start_grid(
            lower=new_lower,
            upper=new_upper,
            count=self.current_grid_num or GRID_COUNT,
            mode=self.current_mode or GRID_MODE,
        )
        return resp

    # ─── 주문 관리 ───────────────────────────────────────────

    def _cancel_pending_orders(self) -> dict:
        """미체결 주문 전체 취소."""
        try:
            orders_resp = self._get(
                "/api/v5/trade/orders-pending",
                params={"instId": SYMBOL, "ordType": "limit"}
            )
            orders = orders_resp.get("data", [])
            if not isinstance(orders, list):
                self._log(f"미체결 주문 조회 응답 구조 이상: {type(orders)}", level="ERROR")
                return {"status": "error", "msg": "unexpected response structure"}
            if not orders:
                return {"status": "no_pending_orders"}

            cancel_list = []
            for o in orders:
                if isinstance(o, dict) and "ordId" in o:
                    cancel_list.append({"instId": SYMBOL, "ordId": o["ordId"]})
            if not cancel_list:
                return {"status": "no_pending_orders"}

            for i in range(0, len(cancel_list), 20):
                batch = cancel_list[i:i+20]
                self._post("/api/v5/trade/cancel-batch-orders", batch)

            self._log(f"미체결 주문 {len(cancel_list)}개 취소 완료")
            return {"status": "cancelled", "count": len(cancel_list)}
        except Exception as e:
            self._log(f"미체결 주문 취소 실패: {e}", level="ERROR")
            return {"status": "error", "msg": str(e)}

    # ─── OKX API 서명 & 호출 ─────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        try:
            msg    = timestamp + method + path + body
            digest = hmac.new(OKX_SECRET_KEY.encode(), msg.encode(), hashlib.sha256).digest()
            return base64.b64encode(digest).decode()
        except Exception as e:
            self._log(f"HMAC 서명 실패 (키가 유효하지 않을 수 있음): {e}", level="ERROR")
            raise

    def _headers(self, method: str, path: str, body: str = "") -> dict:
        now = datetime.now(timezone.utc)
        ts  = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        sig = self._sign(ts, method, path, body)
        return {
            "OK-ACCESS-KEY":        OKX_API_KEY,
            "OK-ACCESS-SIGN":       sig,
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
            "Content-Type":         "application/json",
            **({"x-simulated-trading": "1"} if DEMO_MODE else {}),
        }

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                headers = self._headers("POST", path, body_str)
                r = self.client.post(path, content=body_str, headers=headers)
                try:
                    return r.json()
                except (json.JSONDecodeError, ValueError) as e:
                    self._log(f"POST {path} JSON 파싱 실패 (시도 {attempt}/{_MAX_RETRIES}): {e}", level="ERROR")
                    last_err = e
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                self._log(f"POST {path} 네트워크 오류 (시도 {attempt}/{_MAX_RETRIES}): {e}", level="ERROR")
                last_err = e
            except Exception as e:
                self._log(f"POST {path} 실패: {e}", level="ERROR")
                return {"code": "-1", "msg": str(e)}
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
        return {"code": "-1", "msg": f"max retries exceeded: {last_err}"}

    def _get(self, path: str, params: dict = None) -> dict:
        query = ""
        if params:
            query = "?" + "&".join(f"{k}={v}" for k, v in params.items())
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                headers = self._headers("GET", path + query)
                r = self.client.get(path, params=params, headers=headers)
                try:
                    return r.json()
                except (json.JSONDecodeError, ValueError) as e:
                    self._log(f"GET {path} JSON 파싱 실패 (시도 {attempt}/{_MAX_RETRIES}): {e}", level="ERROR")
                    last_err = e
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                self._log(f"GET {path} 네트워크 오류 (시도 {attempt}/{_MAX_RETRIES}): {e}", level="ERROR")
                last_err = e
            except Exception as e:
                self._log(f"GET {path} 실패: {e}", level="ERROR")
                return {"code": "-1", "msg": str(e)}
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
        return {"code": "-1", "msg": f"max retries exceeded: {last_err}"}

    # ─── 로깅 ────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] [GridController] {msg}")
