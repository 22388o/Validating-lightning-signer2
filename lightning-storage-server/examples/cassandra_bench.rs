use lightning_storage_server::database::cassandra;
use lightning_storage_server::database::scylla;
use lightning_storage_server::{Database, Value};
use std::sync::Arc;
use std::time::Instant;

const ITEMS_PER_TX: u32 = 64;
const CONCURRENT_TASKS: u32 = 8;
const ROUNDS: u32 = 128;

#[tokio::main]
async fn main() {
    // cdrs-tokio driver:
    let db = cassandra::new_and_clear().await.unwrap();
    // scylla driver:
    // let db = scylla::new_and_clear().await.unwrap();
    let start = Instant::now();
    for o in 0..ROUNDS {
        let mut tasks = Vec::new();
        for i in 0..CONCURRENT_TASKS {
            tasks.push(tokio::spawn(do_insert(db.clone(), o * 100 + i)));
        }

        for task in tasks {
            task.await.unwrap();
        }
    }
    let end = Instant::now();
    let elapsed_ms = (end - start).as_millis() as u32;
    println!("done in {} ms", elapsed_ms);
    println!("{} inserts per second", ROUNDS * CONCURRENT_TASKS * ITEMS_PER_TX * 1000 / elapsed_ms);
}

async fn do_insert(db: Arc<dyn Database>, i: u32) {
    let client_id = [0x01];
    let start = Instant::now();
    let kvs = (0..ITEMS_PER_TX)
        .map(|j| {
            (format!("{}-{}", i, j), Value { version: 0, value: [(j % 256) as u8; 128].to_vec() })
        })
        .collect();
    db.put(&client_id, &kvs).await.unwrap();
    let end = Instant::now();
    println!("{} in {} ms", i, (end - start).as_millis());
}
