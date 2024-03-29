# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a deep Q-networks (DQN) agent."""

from dqn.actor import behavior_policy
from dqn.actor import default_behavior_policy
from dqn.agent import DQN, DQNEval
from dqn.alternating_eps_agent import AltDQN, AltDQNEval
from dqn.builder import DQNBuilder
from dqn.config import DQNConfig
from dqn.learning import DQNLearner
from dqn.learning_lib import SGDLearner
from dqn.losses import PrioritizedDoubleQLearning
from dqn.losses import QrDqn
