use super::{Database, Error as DatabaseError};
use crate::Value;
use async_trait::async_trait;
use scylla::batch::Batch;
use scylla::{IntoTypedRows, Session, SessionBuilder};
use std::sync::Arc;
use thiserror::Error;
use tokio::time::Instant;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("new session error: {0}")]
    NewSessionError(#[from] scylla::transport::errors::NewSessionError),
    #[error("query error: {0}")]
    QueryError(#[from] scylla::transport::errors::QueryError),
}

pub struct ScyllaDatabase {
    session: Session,
}

pub async fn new() -> Result<Arc<dyn Database>, DatabaseError> {
    let session = ScyllaDatabase::make_session().await?;
    Ok(Arc::new(ScyllaDatabase { session }))
}

pub async fn new_and_clear() -> Result<Arc<dyn Database>, DatabaseError> {
    let session = ScyllaDatabase::make_session().await?;
    session.query("TRUNCATE ks.data", ()).await.map_err::<DbError, _>(|e| e.into())?;
    Ok(Arc::new(ScyllaDatabase { session }))
}

impl ScyllaDatabase {
    async fn make_session() -> Result<Session, DatabaseError> {
        let uri = std::env::var("CASSANDRA").unwrap_or("172.17.0.2:9042".to_string());

        println!("Connecting to {} ...", uri);

        let session: Session = SessionBuilder::new()
            .known_node(uri)
            .build()
            .await
            .map_err::<DbError, _>(|e| e.into())?;
        Self::migrate_database(&session).await?;
        Ok(session)
    }

    async fn migrate_database(session: &Session) -> Result<(), DatabaseError> {
        session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 3}", &[]).await
            .map_err::<DbError, _>(|e| e.into())?;
        session
            .query(
                "CREATE TABLE IF NOT EXISTS ks.data (client blob, key varchar, version bigint, value blob, primary key (client, key))",
                &[],
            )
            .await.map_err::<DbError, _>(|e| e.into())?;
        Ok(())
    }
}

#[async_trait]
impl Database for ScyllaDatabase {
    async fn put(&self, client_id: &[u8], kvs: &Vec<(String, Value)>) -> Result<(), DatabaseError> {
        let mut batch: Batch = Default::default();
        let prepared = self
            .session
            .prepare("INSERT INTO ks.data (client, key, version, value) VALUES (?, ?, ?, ?) IF NOT EXISTS")
            // .prepare("UPDATE ks.data SET version = ?, value = ? WHERE client = ? AND key = ? IF version = ?")
            // .prepare("UPDATE ks.data SET version = ?, value = ? WHERE client = ? AND key = ? IF version = ?")
            .await
            .map_err::<DbError, _>(|e| e.into())?;
        let mut query_args = Vec::new();
        for (key, value) in kvs {
            batch.append_statement(prepared.clone());
            // we lose one bit here because cassandra only does signed integers
            let version = value.version as i64;
            // let prev_version = if version == 0 { None } else { Some(version - 1) };
            // query_args.push((version, value.value.to_vec(), client_id.to_vec(), key, prev_version));
            query_args.push((client_id.to_vec(), key, version, value.value.to_vec()));
        }

        for _ in 0..5 {
            let start = Instant::now();
            let result =
                self.session.batch(&batch, &query_args).await.map_err::<DbError, _>(|e| e.into());
            if result.is_ok() {
                break;
            }
            let end = Instant::now();
            println!(
                "Retrying put after {} ms due to {:?}",
                (end - start).as_millis(),
                result.err()
            );
        }
        // TODO check result for conflicts
        Ok(())
    }

    async fn get_with_prefix(
        &self,
        client_id: &[u8],
        key_prefix: String,
    ) -> Result<Vec<(String, Value)>, DatabaseError> {
        let rows = self.session
            .query(
                "SELECT key, version, value FROM ks.data WHERE client = ? AND key >= ? ORDER by key ASC",
                (client_id.to_vec(), &key_prefix),
            )
            .await
            .map_err::<DbError, _>(|e| e.into())?
            .rows;
        if let Some(rows) = rows {
            let result = rows
                .into_typed::<(String, i64, Vec<u8>)>()
                .map(|row| {
                    let (key, version, value) = row.unwrap();
                    (key, Value { version: version as u64, value })
                })
                .filter(|(k, _)| k.starts_with(&key_prefix))
                .collect();
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }
}
