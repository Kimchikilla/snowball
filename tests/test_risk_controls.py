import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config
from cost_guard import RecoveryCascade
from main_agent import GridAgent
from grid_controller import GridController
from market_analyzer import MarketSignal
from multi_agent import AgentOpinion, MultiAgentJudge


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class FakeFillController:
    bot_id = "bot"

    def __init__(self, fills):
        self.fills = fills

    def _get(self, path, params=None):
        return {"data": self.fills}


class FakeGridController(GridController):
    def __init__(self):
        self.bot_id = "bot"
        self.current_lower = 2000.0
        self.current_upper = 2500.0
        self.current_grid_num = 10
        self.current_mode = "arithmetic"
        self.started = None

    def stop_grid(self, sell_remaining=False):
        self.bot_id = None
        return {"code": "0"}

    def start_grid(self, lower=None, upper=None, count=None, mode=None):
        self.current_lower = float(lower)
        self.current_upper = float(upper)
        self.current_grid_num = int(count)
        self.current_mode = mode
        self.started = (lower, upper, count, mode)
        self.bot_id = "new"
        return {"code": "0"}

    def _log(self, msg, level="INFO"):
        pass


class QueryFailGridController(FakeGridController):
    def __init__(self):
        super().__init__()
        self.bot_id = None
        self.start_called = False

    def sync_existing_bot(self):
        return {"status": "query_failed", "msg": "timeout"}

    def start_grid(self, lower=None, upper=None, count=None, mode=None):
        self.start_called = True
        return super().start_grid(lower, upper, count, mode)


def make_agent():
    agent = object.__new__(GridAgent)
    agent.holding_qty = 0.0
    agent.holding_cost = 0.0
    agent.realized_pnl = 0.0
    agent.daily_realized = 0.0
    agent.daily_fees = 0.0
    agent.daily_buys = 0
    agent.daily_sells = 0
    agent.daily_buy_vol = 0.0
    agent.daily_sell_vol = 0.0
    agent.daily_buy_cost = 0.0
    agent.daily_sell_revenue = 0.0
    agent.total_fees_paid = 0.0
    agent.grid_restart_times = []
    agent.grid_restart_count = 0
    agent.entry_price = None
    agent.last_fill_id = "old"
    agent.notifier = FakeNotifier()
    agent._log = lambda *args, **kwargs: None
    return agent


def make_signal(
    risk_score=30.0,
    atr_current=10.0,
    atr_avg=10.0,
    volume_ratio=1.0,
    trend="SIDEWAYS",
    trend_strength=0.0,
):
    return MarketSignal(
        risk_score=risk_score,
        atr_score=0.0,
        rsi_score=0.0,
        bb_score=0.0,
        volume_score=0.0,
        atr_current=atr_current,
        atr_avg=atr_avg,
        rsi=50.0,
        bb_width=1.0,
        volume_ratio=volume_ratio,
        trend=trend,
        trend_strength=trend_strength,
        ema_short=2_100.0,
        ema_long=2_100.0,
        adx=trend_strength,
        state="NORMAL",
        reason="test",
    )


class RiskControlTests(unittest.TestCase):
    def setUp(self):
        self.old_max_loss = config.MAX_LOSS_PERCENT
        self.old_grid_budget = config.GRID_BUDGET
        self.old_trigger_score = config.LLM_TRIGGER_SCORE
        config.MAX_LOSS_PERCENT = 15.0
        config.GRID_BUDGET = 38_000.0
        config.LLM_TRIGGER_SCORE = 55

    def tearDown(self):
        config.MAX_LOSS_PERCENT = self.old_max_loss
        config.GRID_BUDGET = self.old_grid_budget
        config.LLM_TRIGGER_SCORE = self.old_trigger_score

    def test_stop_loss_uses_average_cost_before_stale_entry_price(self):
        agent = make_agent()
        agent.holding_qty = 10.0
        agent.holding_cost = 22_000.0
        agent.entry_price = 1_000.0

        status = agent._stop_loss_status(1_870.0)

        self.assertTrue(status["triggered"])
        self.assertEqual(status["basis"], "avg_cost")
        self.assertAlmostEqual(status["price_loss_pct"], 15.0)

    def test_buy_fee_is_added_to_cost_basis(self):
        agent = make_agent()
        agent.controller = FakeFillController([
            {"ordId": "new", "side": "buy", "px": "100", "sz": "1", "fee": "-0.001"},
            {"ordId": "old"},
        ])

        agent._check_fills(current_price=100.0)

        self.assertEqual(agent.holding_qty, 1.0)
        self.assertAlmostEqual(agent.holding_cost, 100.1)
        self.assertAlmostEqual(agent.total_fees_paid, 0.1)

    def test_widen_grid_never_shrinks_current_range(self):
        controller = FakeGridController()

        controller.widen_grid(atr_value=2.0, current_price=2_100.0)

        new_range = controller.current_upper - controller.current_lower
        self.assertGreaterEqual(new_range, 625.0)
        self.assertEqual(controller.current_grid_num, 10)
        self.assertEqual(controller.current_mode, "arithmetic")

    def test_grid_lookup_failure_does_not_start_duplicate_bot(self):
        controller = QueryFailGridController()

        result = controller.ensure_grid_running()

        self.assertFalse(controller.start_called)
        self.assertEqual(result["status"], "sync_failed")

    def test_restart_guard_blocks_widen_when_losing_without_realized_edge(self):
        agent = make_agent()
        agent.holding_qty = 10.0
        agent.holding_cost = 22_000.0
        agent.realized_pnl = -100.0

        allowed, reason = agent._check_restart_allowed("WIDEN", 2_100.0)

        self.assertFalse(allowed)
        self.assertIn("WIDEN", reason)

    def test_agent_events_ignore_grid_boundary_without_real_risk(self):
        agent = make_agent()
        signal = make_signal(risk_score=30.0)

        events = agent._detect_events(signal, price=2_450.0)

        self.assertEqual(events, [])

    def test_agent_events_use_configurable_trigger_score(self):
        agent = make_agent()
        config.LLM_TRIGGER_SCORE = 70

        quiet = agent._detect_events(make_signal(risk_score=69.9), price=2_100.0)
        triggered = agent._detect_events(make_signal(risk_score=70.0), price=2_100.0)

        self.assertEqual(quiet, [])
        self.assertTrue(any("리스크 스코어" in event for event in triggered))

    def test_recovery_fallback_never_restarts_or_stops_from_score_only(self):
        self.assertEqual(
            RecoveryCascade.rule_based_fallback(95.0, "BEARISH", 40.0),
            "MAINTAIN",
        )


class MultiAgentConsensusTests(unittest.TestCase):
    def setUp(self):
        self.judge = object.__new__(MultiAgentJudge)

    def opinion(self, role, action, confidence=8, reason="test"):
        return AgentOpinion(role=role, action=action, confidence=confidence, reason=reason)

    def test_any_maintain_blocks_change(self):
        action, reason = self.judge._coordinate([
            self.opinion("operator", "WIDEN", 9),
            self.opinion("critic", "MAINTAIN", 8),
        ])

        self.assertEqual(action, "MAINTAIN")
        self.assertIn("MAINTAIN", reason)

    def test_matching_change_requires_confidence_above_seven(self):
        low_action, _ = self.judge._coordinate([
            self.opinion("operator", "WIDEN", 7),
            self.opinion("critic", "WIDEN", 7),
        ])
        high_action, _ = self.judge._coordinate([
            self.opinion("operator", "WIDEN", 9),
            self.opinion("critic", "WIDEN", 8),
        ])

        self.assertEqual(low_action, "MAINTAIN")
        self.assertEqual(high_action, "WIDEN")

    def test_stop_requires_systemic_reason_and_high_confidence(self):
        weak_action, _ = self.judge._coordinate([
            self.opinion("operator", "STOP", 9, "위험해 보임"),
            self.opinion("critic", "STOP", 9, "보수적으로 정지"),
        ])
        systemic_action, _ = self.judge._coordinate([
            self.opinion("operator", "STOP", 9, "거래소 장애 발생"),
            self.opinion("critic", "STOP", 9, "시스템적 시장 붕괴"),
        ])

        self.assertEqual(weak_action, "MAINTAIN")
        self.assertEqual(systemic_action, "STOP")


if __name__ == "__main__":
    unittest.main()
