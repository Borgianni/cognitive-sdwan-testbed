import tempfile
import unittest
from pathlib import Path
import numpy as np
from sdwan_rl.agents.expected_sarsa import ExpectedSarsaAgent
from sdwan_rl.config import RLConfig
from sdwan_rl.learning_agents import save_learner,load_learner
from sdwan_rl.policy_audit import weights_digest
from sdwan_rl.telemetry_core import TelemetryConfig


class ExpectedSarsaTests(unittest.TestCase):
    def test_expected_backup_and_update(self):
        a=ExpectedSarsaAgent(RLConfig(gamma=.5,eps_start=.2),seed=1,alpha=.1,state_size=1)
        a.weights[:]=[[1.,2.],[0.,0.]]
        a.remember([1],0,2,[1],False)
        out=a.train_step()
        self.assertAlmostEqual(out["td_target"],2+.5*1.9)
        self.assertAlmostEqual(a.weights[0,0],1.195)
        self.assertAlmostEqual(a.weights[1,0],.195)
        self.assertEqual(a.weights[0,1],2)
        self.assertEqual(a.updates,1)
        self.assertEqual(len(a.memory),0)

    def test_terminal_target_and_frozen(self):
        a=ExpectedSarsaAgent(RLConfig(),seed=1)
        s=np.ones(6)
        a.remember(s,1,3,s,True)
        self.assertEqual(a.train_step()["td_target"],3)
        a.freeze()
        digest=weights_digest(a)
        a.remember(s,0,9,s,False)
        self.assertIsNone(a.train_step())
        self.assertEqual(a.traced_action(s)[0],int(np.argmax(a.q_values(s))))
        self.assertEqual(a.eps,0)
        self.assertEqual(weights_digest(a),digest)

    def test_seeds_and_shapes(self):
        a,b,c=[ExpectedSarsaAgent(RLConfig(),seed=i) for i in (1,1,2)]
        self.assertEqual(weights_digest(a),weights_digest(b))
        self.assertNotEqual(weights_digest(a),weights_digest(c))
        self.assertEqual([a.act(np.zeros(6)) for _ in range(10)],[b.act(np.zeros(6)) for _ in range(10)])
        with self.assertRaises(ValueError):
            a.q_values([1]*5)

    def test_roundtrip_and_mismatch(self):
        cfg=RLConfig(state_representation="continuous")
        t=TelemetryConfig().measurement_settings()
        protocol={"split":"diagnostic"}
        a=ExpectedSarsaAgent(cfg,seed=5)
        a.remember(np.zeros(6),0,1,np.ones(6),False)
        a.train_step()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"model.npz"
            save_learner("expected_sarsa",a,cfg,p,protocol,5,t)
            b=ExpectedSarsaAgent(cfg,seed=5)
            load_learner("expected_sarsa",b,cfg,p,protocol,5,t)
            self.assertTrue(b.frozen)
            self.assertEqual(weights_digest(a),weights_digest(b))
            with self.assertRaises(ValueError):
                load_learner("expected_sarsa",ExpectedSarsaAgent(cfg,alpha=.02),cfg,p,protocol,5,t)


if __name__=="__main__":
    unittest.main()
