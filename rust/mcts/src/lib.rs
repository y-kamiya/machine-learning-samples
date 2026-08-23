use maze::Field;

pub fn create_field(is_print: bool) -> Field {
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

