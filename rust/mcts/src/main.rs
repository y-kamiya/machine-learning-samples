use rand::prelude::*;
use maze::{Action, Field, Pos, NodeType};

fn simulate(max: i32) -> f32 {
    let mut count = 0;
    let mut rng = rand::rng();
    for _ in 0..max {
        let x = rng.random_range(0.0..=1.0);
        let y = rng.random_range(0.0..=1.0);
        let d = x * x + y * y;
        if d <= 1.0 {
            count += 1;
        }
    }
    count as f32 / max as f32 * 4.0
}

fn monte_carlo() {
    for n in [100, 1000, 10000, 100000] {
        let output = simulate(n);
        println!("sample count {n}: {output}");
    }
}

fn monte_carlo_search() {
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

fn create_field(is_print: bool) -> Field {
    let field_sample = [
        "######",
        "#S...#",
        "##.###",
        "#...G#",
        "######",
    ];
    let field = Field::new(&field_sample);
    if is_print {
        println!("{}", field);
    }
    field
}

fn random_maze() {
    let field = create_field(true);
    let mut pos = field.start;
    println!("start pos:{}", pos);
    for i in 0..10 {
        let action = Action::uniform();
        pos = field.act(pos, action);
        println!("i:{}, action:{}, pos:{}", i, action, pos);
    }
}

fn mcts() {
    let field = create_field(true);

    // let tree = 
    for i in 0..10 {
    }


}

fn main() {
    //monte_carlo();
    //monte_carlo_search();
    // random_maze();
    mcts();
}
