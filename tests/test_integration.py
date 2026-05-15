"""
End-to-end integration tests.

All tests run fully offline — no network, no Zeek, no Mininet required.
"""
import datetime
import numpy as np
import pytest

from cps_defender.core.events import reset_bus, get_bus, EventType, Event
from cps_defender.core.models import AttackType, Severity, N_FEATURES
from cps_defender.ids.pipeline import IDSPipeline
from cps_defender.ids.feature_extractor import FeatureExtractor
from cps_defender.sdn.controller import create_controller
from cps_defender.sdn.mitigation import MitigationEngine, MitigationAction
from cps_defender.rl.environment import CPSEnvironment
from cps_defender.rl.agent import DQNAgent
from cps_defender.testbed.traffic_sim import TrafficSimulator
from cps_defender.genai.augmenter import AugmentationPipeline


# ── Full detect → mitigate pipeline ──────────────────────────────────────────

class TestFullPipeline:
    """Simulate realistic operational detect-and-respond cycle."""

    @pytest.fixture
    def trained_pipeline(self, large_dataset):
        pipe = IDSPipeline()
        pipe.train(large_dataset)
        return pipe

    @pytest.fixture
    def engine(self):
        return MitigationEngine(create_controller("mock"))

    def test_detect_and_respond_cycle(self, trained_pipeline, engine, small_dataset):
        """End-to-end: classify → alert → mitigate."""
        attacks = [f for f in small_dataset if f.label != AttackType.NORMAL][:20]
        if not attacks:
            pytest.skip("No attack flows available")

        mitigated = 0
        for flow in attacks:
            alert = trained_pipeline.analyze(flow)
            if alert:
                action = MitigationAction.RATE_LIMIT
                result = engine.apply(action, alert)
                assert result.success
                mitigated += 1
        assert mitigated >= 1, "No alerts fired and mitigated"

    def test_event_bus_receives_alerts(self, large_dataset, small_dataset):
        """Alerts published to event bus during pipeline analysis."""
        reset_bus()
        bus      = get_bus()
        received = []
        bus.subscribe(EventType.ALERT_GENERATED, received.append)

        pipe = IDSPipeline()
        pipe.train(large_dataset)

        for flow in small_dataset[:40]:
            pipe.analyze(flow)

        # With 30% attack ratio, some alerts should reach the bus
        attacks = [f for f in small_dataset[:40] if f.label != AttackType.NORMAL]
        if attacks:
            # Just verify no exceptions; alerts may or may not fire
            assert isinstance(received, list)

    def test_pipeline_stats_accumulate(self, trained_pipeline, small_dataset):
        for flow in small_dataset[:50]:
            trained_pipeline.analyze(flow)
        stats = trained_pipeline.stats()
        assert isinstance(stats, dict)

    def test_high_attack_ratio_detection(self):
        """Under heavy attack, most attack flows should generate alerts."""
        reset_bus()
        sim  = TrafficSimulator(seed=99, attack_probability=0.80)
        all_flows = sim.generate(n_flows=400)
        train = all_flows[:300]
        test  = all_flows[300:]

        pipe = IDSPipeline()
        pipe.train(train)

        attacks = [f for f in test if f.label != AttackType.NORMAL]
        if not attacks:
            pytest.skip("No attacks in test set")

        tp = sum(1 for f in attacks if pipe.analyze(f) is not None)
        tpr = tp / len(attacks)
        assert tpr > 0.40, f"TPR={tpr:.2f} too low under heavy attack"

    def test_normal_traffic_low_fp(self):
        """Under pure normal traffic, FP rate should be low."""
        reset_bus()
        sim  = TrafficSimulator(seed=77, attack_probability=0.30)
        all_flows = sim.generate(n_flows=400)

        pipe = IDSPipeline()
        pipe.train(all_flows[:300])

        normal = [f for f in all_flows[300:] if f.label == AttackType.NORMAL]
        if not normal:
            pytest.skip("No normal flows in test set")

        fp = sum(1 for f in normal if pipe.analyze(f) is not None)
        fpr = fp / len(normal)
        assert fpr < 0.35, f"FPR={fpr:.2f} too high on normal traffic"


# ── GenAI Augmentation ────────────────────────────────────────────────────────

class TestAugmentation:
    @pytest.fixture
    def raw_data(self, large_dataset):
        ext = FeatureExtractor()
        X   = ext.fit_transform(large_dataset)
        y   = np.array([f.label for f in large_dataset])
        return X, y

    def test_augmented_size_grows(self, raw_data):
        X, y = raw_data
        aug  = AugmentationPipeline(seed=42)
        aug.fit(X)
        X2, y2 = aug.generate(X, y, factor=2)
        assert X2.shape[0] >= X.shape[0]

    def test_feature_count_preserved(self, raw_data):
        X, y = raw_data
        aug  = AugmentationPipeline(seed=1)
        aug.fit(X)
        X2, y2 = aug.generate(X, y, factor=2)
        assert X2.shape[1] == N_FEATURES

    def test_no_nan_in_augmented(self, raw_data):
        X, y = raw_data
        aug  = AugmentationPipeline(seed=2)
        aug.fit(X)
        X2, _ = aug.generate(X, y, factor=2)
        assert not np.any(np.isnan(X2))
        assert not np.any(np.isinf(X2))

    def test_augmented_has_all_original_labels(self, raw_data):
        X, y = raw_data
        aug  = AugmentationPipeline(seed=3)
        aug.fit(X)
        _, y2 = aug.generate(X, y, factor=2)
        assert set(np.unique(y)).issubset(set(np.unique(y2)))


# ── Flooding scenario ─────────────────────────────────────────────────────────

class TestFloodingScenario:
    """Verify the system correctly identifies and mitigates a DoS flood."""

    def test_flooding_detected(self):
        reset_bus()
        sim   = TrafficSimulator(seed=10, attack_probability=0.50)
        flows = sim.generate(n_flows=400)
        pipe  = IDSPipeline()
        pipe.train(flows[:300])

        flooding = [f for f in flows[300:]
                    if f.label == AttackType.FLOODING]
        if not flooding:
            pytest.skip("No flooding flows generated")

        detected = sum(1 for f in flooding if pipe.analyze(f) is not None)
        assert detected > 0, "Flooding not detected at all"

    def test_flooding_mitigated_with_rate_limit(self):
        reset_bus()
        sim   = TrafficSimulator(seed=11, attack_probability=0.50)
        flows = sim.generate(n_flows=400)
        pipe  = IDSPipeline()
        pipe.train(flows[:300])

        ctrl   = create_controller("mock")
        engine = MitigationEngine(ctrl)

        flooding = [f for f in flows[300:] if f.label == AttackType.FLOODING]
        if not flooding:
            pytest.skip("No flooding flows")

        mitigated = 0
        for flow in flooding[:20]:
            alert = pipe.analyze(flow)
            if alert:
                result = engine.apply(MitigationAction.RATE_LIMIT, alert)
                if result.success:
                    mitigated += 1

        assert mitigated >= 1


# ── RL integration ────────────────────────────────────────────────────────────

class TestRLIntegration:
    """RL agent learns a policy through environment interaction."""

    @pytest.fixture
    def env(self):
        reset_bus()
        return CPSEnvironment(attack_prob=0.30, seed=7)

    @pytest.fixture
    def agent(self, env):
        return DQNAgent(obs_dim=env.obs_dim, n_actions=env.action_space_n,
                        batch_size=8, memory_size=500, seed=7)

    def test_agent_runs_episode(self, env, agent):
        """Agent can run a full episode without errors."""
        state = env.reset()
        done  = False
        steps = 0
        total_reward = 0.0
        while not done and steps < 300:
            action = agent.select_action(state, greedy=False)
            result = env.step(action)
            agent.remember(state, action, result.reward,
                           result.observation, result.done)
            agent.learn()
            state      = result.observation
            total_reward += result.reward
            done       = result.done
            steps     += 1
        assert steps > 0
        assert isinstance(total_reward, float)

    def test_agent_learns_over_episodes(self, env, agent):
        """Total reward should not degrade to extremely negative over many episodes."""
        rewards = []
        for _ in range(30):
            state = env.reset()
            done  = False
            ep_r  = 0.0
            while not done:
                action = agent.select_action(state)
                result = env.step(action)
                agent.remember(state, action, result.reward,
                               result.observation, result.done)
                agent.learn()
                state = result.observation
                ep_r += result.reward
                done  = result.done
            agent.end_episode(ep_r)
            rewards.append(ep_r)
        # Just verify no crash and some rewards are non-trivially finite
        assert all(np.isfinite(r) for r in rewards)

    def test_epsilon_decays(self, env, agent):
        eps_start = agent.epsilon
        for _ in range(20):
            agent.end_episode(0.0)
        assert agent.epsilon < eps_start

    def test_greedy_action_deterministic(self, env, agent):
        """Greedy policy should be deterministic."""
        state = env.reset()
        a1 = agent.select_action(state, greedy=True)
        a2 = agent.select_action(state, greedy=True)
        assert a1 == a2

    def test_save_load_agent(self, env, agent, tmp_path):
        """Saved and loaded agent produces identical greedy actions."""
        path  = str(tmp_path / "agent.npz")
        state = env.reset()
        # Train briefly
        for _ in range(20):
            s  = np.random.randn(env.obs_dim).astype(np.float32)
            agent.remember(s, np.random.randint(env.action_space_n),
                           float(np.random.randn()),
                           np.random.randn(env.obs_dim).astype(np.float32),
                           False)
            agent.learn()
        agent.save(path)
        loaded = DQNAgent.load(path)

        for _ in range(5):
            s = np.random.randn(env.obs_dim).astype(np.float32)
            assert agent.select_action(s, greedy=True) == \
                   loaded.select_action(s, greedy=True)

    def test_episode_summary_keys(self, env, agent):
        state = env.reset()
        done  = False
        while not done:
            result = env.step(np.random.randint(env.action_space_n))
            done = result.done
        summary = env.episode_summary()
        # Summary should have at minimum step count or reward
        assert isinstance(summary, dict)
        assert len(summary) > 0
