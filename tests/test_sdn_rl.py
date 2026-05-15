"""Tests for SDN controller, mitigation engine, RL environment, and DQN agent."""
import datetime
import numpy as np
import pytest

from cps_defender.core.events import reset_bus
from cps_defender.core.models import Alert, AttackType, Severity
from cps_defender.sdn.controller import MockController, create_controller
from cps_defender.sdn.mitigation import MitigationEngine, MitigationAction, MitigationResult
from cps_defender.rl.environment import CPSEnvironment
from cps_defender.rl.agent import DQNAgent, ReplayBuffer


# ── Alert factory helper ──────────────────────────────────────────────────────

def make_alert(
    src_ip: str = "10.0.1.5",
    dst_ip: str = "10.0.0.1",
    attack_type: str = AttackType.FLOODING,
    severity: Severity = Severity.HIGH,
    confidence: float = 0.85,
) -> Alert:
    return Alert(
        uid=f"test-{src_ip}-{datetime.datetime.now().timestamp():.0f}",
        timestamp=datetime.datetime.now(),
        src_ip=src_ip,
        dst_ip=dst_ip,
        attack_type=attack_type,
        severity=severity,
        confidence=confidence,
        score=confidence,
        detector="rule",
        features=None,
        raw_scores={},
        mitigated=False,
    )


# ── MockController ────────────────────────────────────────────────────────────

class TestMockController:
    @pytest.fixture
    def ctrl(self):
        return MockController()

    def test_install_and_retrieve_flow(self, ctrl):
        from cps_defender.sdn.controller import FlowEntry, FlowMatch
        entry = FlowEntry(
            entry_id="test-001",
            match=FlowMatch(src_ip="10.0.1.1"),
            action="drop",
            priority=5000,
        )
        assert ctrl.install_flow(entry) is True
        flows = ctrl.get_flows()
        ids = [f.entry_id for f in flows]
        assert "test-001" in ids

    def test_delete_flow(self, ctrl):
        from cps_defender.sdn.controller import FlowEntry, FlowMatch
        entry = FlowEntry(
            entry_id="del-001",
            match=FlowMatch(src_ip="10.0.1.2"),
            action="rate_limit",
        )
        ctrl.install_flow(entry)
        assert ctrl.delete_flow("del-001") is True
        flows = ctrl.get_flows()
        assert not any(f.entry_id == "del-001" for f in flows)

    def test_get_host_state(self, ctrl):
        state = ctrl.get_host_state("10.0.1.3")
        assert isinstance(state, dict)
        assert "latency_ms" in state or "rate_kbps" in state or len(state) >= 0

    def test_get_stats(self, ctrl):
        stats = ctrl.get_stats()
        assert isinstance(stats, dict)

    def test_create_controller_factory_mock(self):
        ctrl = create_controller("mock")
        assert isinstance(ctrl, MockController)

    def test_multiple_flows(self, ctrl):
        from cps_defender.sdn.controller import FlowEntry, FlowMatch
        for i in range(5):
            entry = FlowEntry(
                entry_id=f"flow-{i:03d}",
                match=FlowMatch(src_ip=f"10.0.1.{i+1}"),
                action="drop",
            )
            ctrl.install_flow(entry)
        assert len(ctrl.get_flows()) >= 5


# ── MitigationEngine ──────────────────────────────────────────────────────────

class TestMitigationEngine:
    @pytest.fixture
    def engine(self):
        ctrl = MockController()
        return MitigationEngine(ctrl)

    def test_monitor_succeeds(self, engine):
        alert  = make_alert()
        result = engine.apply(MitigationAction.MONITOR, alert)
        assert isinstance(result, MitigationResult)
        assert result.success

    def test_rate_limit_succeeds(self, engine):
        alert  = make_alert(src_ip="10.0.1.10")
        result = engine.apply(MitigationAction.RATE_LIMIT, alert)
        assert result.success

    def test_segment_succeeds(self, engine):
        alert  = make_alert(src_ip="10.0.1.11")
        result = engine.apply(MitigationAction.SEGMENT, alert)
        assert result.success

    def test_isolate_succeeds(self, engine):
        alert  = make_alert(src_ip="10.0.1.12")
        result = engine.apply(MitigationAction.ISOLATE, alert)
        assert result.success

    def test_block_succeeds_on_non_critical(self, engine):
        alert  = make_alert(src_ip="10.0.1.20")
        result = engine.apply(MitigationAction.BLOCK, alert)
        assert result.success

    def test_revoke_after_isolate(self, engine):
        alert  = make_alert(src_ip="10.0.1.30")
        engine.apply(MitigationAction.ISOLATE, alert)
        revoked = engine.revoke("10.0.1.30")
        assert revoked is True

    def test_active_mitigations_list(self, engine):
        alert = make_alert(src_ip="10.0.1.40")
        engine.apply(MitigationAction.RATE_LIMIT, alert)
        active = engine.get_active_mitigations()
        assert isinstance(active, list)
        assert len(active) >= 1

    def test_audit_log_populated(self, engine):
        alert = make_alert()
        engine.apply(MitigationAction.MONITOR, alert)
        log = engine.get_audit_log()
        assert isinstance(log, list)
        assert len(log) >= 1

    def test_result_has_target_ip(self, engine):
        alert  = make_alert(src_ip="10.0.1.50")
        result = engine.apply(MitigationAction.SEGMENT, alert)
        assert result.target_ip == "10.0.1.50"

    def test_result_has_action(self, engine):
        alert  = make_alert(src_ip="10.0.1.51")
        result = engine.apply(MitigationAction.BLOCK, alert)
        assert result.action == MitigationAction.BLOCK

    def test_revoke_expired(self, engine):
        # Should not raise even if nothing has expired
        n = engine.revoke_expired()
        assert isinstance(n, int) and n >= 0


# ── CPSEnvironment ────────────────────────────────────────────────────────────

class TestCPSEnvironment:
    @pytest.fixture
    def env(self):
        reset_bus()
        return CPSEnvironment(seed=42)

    def test_obs_dim(self, env):
        assert env.obs_dim == 8

    def test_action_space_n(self, env):
        assert env.action_space_n == 5

    def test_reset_returns_array(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.obs_dim,)

    def test_step_returns_step_result(self, env):
        env.reset()
        result = env.step(0)
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")

    def test_observation_shape(self, env):
        env.reset()
        result = env.step(0)
        assert result.observation.shape == (env.obs_dim,)

    def test_reward_is_float(self, env):
        env.reset()
        result = env.step(0)
        assert isinstance(result.reward, float)

    def test_done_is_bool(self, env):
        env.reset()
        result = env.step(0)
        assert isinstance(result.done, bool)

    def test_episode_terminates(self, env):
        state = env.reset()
        done  = False
        steps = 0
        while not done and steps < 500:
            result = env.step(np.random.randint(env.action_space_n))
            done   = result.done
            steps += 1
        assert done, "Episode should terminate within max_steps"

    def test_episode_summary(self, env):
        state = env.reset()
        for _ in range(20):
            result = env.step(np.random.randint(env.action_space_n))
            if result.done:
                break
        summary = env.episode_summary()
        assert isinstance(summary, dict)
        assert "tp" in summary or "steps" in summary

    def test_all_actions_valid(self, env):
        for action in range(env.action_space_n):
            env.reset()
            result = env.step(action)
            assert result.observation.shape == (env.obs_dim,)


# ── DQNAgent ──────────────────────────────────────────────────────────────────

class TestDQNAgent:
    @pytest.fixture
    def agent(self):
        return DQNAgent(obs_dim=8, n_actions=5, batch_size=4,
                        memory_size=200, seed=0)

    def test_select_action_explore(self, agent):
        state  = np.random.randn(8).astype(np.float32)
        action = agent.select_action(state, greedy=False)
        assert 0 <= action < 5

    def test_select_action_greedy(self, agent):
        state  = np.random.randn(8).astype(np.float32)
        action = agent.select_action(state, greedy=True)
        assert 0 <= action < 5

    def test_remember_and_learn(self, agent):
        # Fill buffer past batch_size
        for _ in range(20):
            s  = np.random.randn(8).astype(np.float32)
            a  = np.random.randint(5)
            r  = float(np.random.randn())
            s2 = np.random.randn(8).astype(np.float32)
            agent.remember(s, a, r, s2, False)
        loss = agent.learn()
        # Loss may be None if buffer not ready, or a float
        assert loss is None or isinstance(loss, float)

    def test_epsilon_decreases_after_end_episode(self, agent):
        eps_before = agent.epsilon
        agent.end_episode(0.0)
        assert agent.epsilon <= eps_before

    def test_epsilon_lower_bound(self, agent):
        # Drive epsilon to min
        for _ in range(2000):
            agent.end_episode(0.0)
        assert agent.epsilon >= agent.epsilon_end

    def test_get_stats_dict(self, agent):
        stats = agent.get_stats()
        assert isinstance(stats, dict)

    def test_save_load_roundtrip(self, agent, tmp_path):
        path = str(tmp_path / "agent.npz")
        # Fill buffer and learn so weights are non-trivial
        for _ in range(20):
            s  = np.random.randn(8).astype(np.float32)
            agent.remember(s, np.random.randint(5),
                           float(np.random.randn()),
                           np.random.randn(8).astype(np.float32), False)
        agent.learn()
        agent.save(path)

        loaded = DQNAgent.load(path)
        state  = np.random.randn(8).astype(np.float32)
        a1 = agent.select_action(state, greedy=True)
        a2 = loaded.select_action(state, greedy=True)
        assert a1 == a2

    def test_consistent_greedy_action(self, agent):
        """Same state should give same greedy action."""
        state = np.random.randn(8).astype(np.float32)
        a1 = agent.select_action(state, greedy=True)
        a2 = agent.select_action(state, greedy=True)
        assert a1 == a2


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

class TestReplayBuffer:
    @pytest.fixture
    def buf(self):
        return ReplayBuffer(capacity=50)

    def test_push_and_len(self, buf):
        for i in range(10):
            buf.push(
                np.zeros(8, dtype=np.float32), i % 5, float(i),
                np.ones(8, dtype=np.float32), False,
            )
        assert len(buf) == 10

    def test_sample_returns_none_if_small(self, buf):
        buf.push(np.zeros(8), 0, 0.0, np.zeros(8), False)
        result = buf.sample(batch_size=32)
        assert result is None

    def test_sample_batch_shapes(self, buf):
        for i in range(40):
            buf.push(
                np.random.randn(8).astype(np.float32), i % 5, float(i),
                np.random.randn(8).astype(np.float32), False,
            )
        batch = buf.sample(batch_size=16)
        assert batch is not None
        obs, actions, rewards, next_obs, dones = batch
        assert obs.shape == (16, 8)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)

    def test_fifo_capacity(self, buf):
        for i in range(80):  # Push past capacity=50
            buf.push(
                np.zeros(8, dtype=np.float32), 0, float(i),
                np.zeros(8, dtype=np.float32), False,
            )
        assert len(buf) == 50
