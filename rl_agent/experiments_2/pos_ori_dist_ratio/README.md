# What is the ideal starting position and orientation distance ratio?
If an object is oriented at 0 degrees, it takes a shorter distance to push it to the goal.
There is a minimum distance necessary to push the object to the goal for a given orientation.
We assume that the relation between the distance and the orientation is affine.
min_dist = a * orientation + b

Therefore, max_pos_step = a * max_ori_step + b
This is not perfect, because the actual pos step in the linear interpolation may be smaller than max_pos_step.
However, is the ratio maintained? As the dist is divided, the orientation will also be. * Demonstrate later.


real_pos_step = max_pos_step/r 
real_ori_step = max_ori_step/r
We want to show that real_pos_step >= a * real_ori_step + b
Assuming, max_pos_step >= a * max_ori_step + b

max_pos_step/r >= a * max_ori_step/r + b
max_pos_step >= a * max_ori_step + b * r
=> Certainly true if r <= 1
Or if max_pos_step has a slack larger than b * (r-1).
Might need to remove b. 

Let's find a. 
Experiment:
- Initial distances: ori rand(0, 360), pos rand(0, 60)
- Plot the 3D graphs of for each object:
    - ori x pos x success_rate
    - ori x pos x episode_steps
- 3000 episodes
- Record:
    - object
    - starting pos and ori
    - success
    - episodes steps


Not good => results are sparse. Let's make a more controlled experiment. Fix the orientation dist and 
obtain a bunch of data points for closer dist => 0 to 30. and rectangles and triangles.
Clear myhead => find the question I want to answer.

My question: can I interpolate with very short steps? And how distance relates to orientation? What
is the minimum distance for performance?
Maybe the problem is in the interpolation: getting to each subgoal adds delay because of fine adjustments.
Current hypothesis: it is slower to get from A, B, C than from A to C directly, because of the fine adjustments.
Specially orientation.
Therefore, I need to use subgoal tolerance different from the final goal. To minimize the fine adjustments.
The ultimate test:

Simple test:
- I want to compare subgoal tolerance and epsiode length.
- Run 100 episodes with fixed pos and ori initial distances => the ones we consider using as step small sizes. And a big one.
- Rectangle and triangle.
- Run with different goal tolerances.
- Final analysis:
    - Compare the average episode length for each tolerance at each initial setup.
    - Answer: What is the efficient of each tolerance? Efficiency = episode_length / init_pos_dist

Idea: Just use distance in the subgoals.
Experiment:
- dist_tol = 2. Vary orientation tolerance: np.pi / 36, /9, /3
- Fix: init_ori_dist: np.pi, and vary init_pos_dist: 
Pick the best ori_tol, and test dist_tol = 3, 4, 5

Simplifying:
- Implement subgoal tolerance.
- Fix the initial distance: pos = 120, ori = np.pi
- Vary the subgoal tolerance X max_step pos and orig
- Goal: find the most efficient subgoal tactic and the tradeoff between subgoal tolerance and efficiency.
Tactics:
- 1 shot: sub_tol = final_tol, step: (130, 2*np.pi)
- 2 shots: sub_tol = (2, np.pi), step: (60, np.pi)
- 4 shots: sub_tol = (2, np.pi), step: (30, np.pi)
- 8 shots: sub_tol = (2, np.pi), step: (15, np.pi)
- 16 shots: sub_tol = (2, np.pi), step: (7.5, np.pi)
- 32 shots: sub_tol = (2, np.pi), step: (3.75, np.pi)
- 64 shots: sub_tol = (2, np.pi), step: (3, np.pi)
200 episodes for each tactic.
Later, evaluate with sub_tol = (3, np.pi)

Eu quero facilitar o treinamento e integrar em uma solução com subgoals.
O treino é feito de um subogal para outro.
Quero que o treino seja muito fácil: dist_ori e dist_pos pequenos.