


# Experiments

## First day:
### exp1:
- REINFORCE algo.
- Was not applying final penalty to all time steps.
- Was able to train moonlander to land, but it did a back and forth thing that got lucky sometimes and did not learn how to stabilize itself.

## March 22/2025:
### exp1:
- carry over:
    - moonlander.
    - REINFORCE algo.
- changes:
    - Heavy penalty to y-pos being off. -= 50 * y_pos
    - Apply final penalty without discounting to all time steps.


#### IS THERE VARIANCE?
It can't learn to go straight down, it always uses the side waying with the side engines.

1. Penalizing landing but with poor x_pos. Still learned to sway.
2. Penalizing being off the center x_pos at intermediate steps. Learned to sway.
3. Using discounted rewards gamma=0.9; Learned to sway but also to hover and never actually land.


