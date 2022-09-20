#[cfg(feature = "cassandra")]
pub mod cassandra;
pub mod scylla;
pub mod sled;

use crate::model::Value;
use thiserror::Error;
use tonic::async_trait;

/// Database errors
#[derive(Debug, Error)]
pub enum Error {
    /// underlying database error
    #[error("database error: {0}")]
    Sled(#[from] ::sled::Error),
    #[cfg(feature = "cassandra")]
    #[error("database error: {0}")]
    Cassandra(#[from] cassandra::CdrsError),
    #[cfg(feature = "cassandra")]
    #[error("database error: {0}")]
    Scylla(#[from] scylla::DbError),
    /// version conflicts detected - existing values are returned
    #[error("put conflict: {0:?}")]
    Conflict(Vec<(String, Option<Value>)>),
}

#[async_trait]
pub trait Database: Send + Sync {
    /// Atomically put a vector of key-values into the database.
    ///
    /// If any of the value versions are not the next version, the entire
    /// transaction is aborted and the error includes the existing values.
    async fn put(&self, client_id: &[u8], kvs: &Vec<(String, Value)>) -> Result<(), Error>;

    /// Get all keys matching a prefix from the database
    async fn get_with_prefix(
        &self,
        client_id: &[u8],
        key_prefix: String,
    ) -> Result<Vec<(String, Value)>, Error>;
}
