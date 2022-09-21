use lightning_storage_server::{Database, Value};
use tempfile;

#[test]
fn test_sled() {
    let dir = tempfile::tempdir().unwrap();
    let db = sled::open(&dir).unwrap();
    db.insert(b"yo!", b"v1").unwrap();
    assert_eq!(&db.get(b"yo!").unwrap().unwrap(), b"v1");
    dir.close().unwrap();
}

fn make_value(v: u8) -> Value {
    Value { version: 0, value: vec![v] }
}

#[test]
fn test_database() {
    let dir = tempfile::tempdir().unwrap();
    println!("tempdir: {}", dir.path().display());
    let db = Database::new(dir.path().to_str().unwrap()).unwrap();
    let client_id = vec![1];
    db.put(
        &client_id,
        &vec![
            ("x1a".to_string(), make_value(10)),
            ("x1b".to_string(), make_value(11)),
            ("x2b".to_string(), make_value(20)),
        ],
    )
    .unwrap();
    let values = db.get_with_prefix(&client_id, "x1".to_string()).unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].1.value, vec![10]);
    assert_eq!(values[1].1.value, vec![11]);
    dir.close().unwrap();
}
