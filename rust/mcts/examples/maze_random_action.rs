use maze::Action;
use mcts::create_field;

fn main() {
    let field = create_field(true);
    let mut pos = field.start;
    println!("start pos:{}", pos);
    for i in 0..10 {
        let action = Action::uniform();
        pos = field.act(pos, action);
        println!("i:{}, action:{}, pos:{}", i, action, pos);
    }
}

