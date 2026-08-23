use rand::prelude::*;

fn main() {
    let max = 10000;

    let actions: [i32; 2] = [-1, 1];
    let field = [1, 0, 0, 0, 0, 1];
    let start_pos = 2;
    let n_max_actions = 4;

    let mut rng = rand::rng();
    let mut first_action_goal_count = [0, 0];
    for _ in 0..max {
        let mut pos = start_pos;
        let mut history = Vec::new();
        for _ in 0..n_max_actions {
            let idx = rng.random_range(0..=1);
            history.push(idx);
            pos += actions[idx];
            if field[pos as usize] == 1 {
                let first_action = history[0];
                first_action_goal_count[first_action] += 1;
                break;
            }
        }
    }
    let l = first_action_goal_count[0] as f32 / max as f32;
    let r = first_action_goal_count[1] as f32 / max as f32;
    println!("goal rate left: {l}, right: {r}");

}
