use crate::channel::ChannelId;
use crate::node::Node;
use crate::sync::Arc;

/// State descriptor
#[derive(Debug, Clone)]
pub enum StateDescriptor {
    /// NodeState
    NodeState,
    /// ChainTrackerState
    ChainTrackerState,
    /// VelocityState
    VelocityState,
    /// ChannelSetup
    ChannelSetup(ChannelId),
    /// ChannelEnforcementState
    ChannelEnforcementState(ChannelId),
}

/// State context management
pub struct StateContext {
    node: Arc<Node>,
    reads: Vec<StateDescriptor>,
    writes: Vec<StateDescriptor>,
}

/// State update data (passed in or returned)
pub struct StateUpdate {
    // TODO - depends on choices
}

impl StateUpdate {
    /// Unmarshal an incoming update from a message
    pub fn extract_from_message(/* TODO _msg: &Message */) -> Self {
        // TODO
        StateUpdate {}
    }

    /// Marshal an outgoing update from the node's state
    pub fn compose_from_node(_node: &Arc<Node>, _writes: &Vec<StateDescriptor>) -> Self {
        // TODO
        StateUpdate {}
    }
}

impl StateContext {
    /// Create a StateContext for a node and an incoming update
    pub fn new(node: Arc<Node>, _incoming_update: StateUpdate) -> Self {
        // TODO - apply the incoming_update to the node

        StateContext { node, reads: vec![], writes: vec![] }
    }

    /// Prepare state for handling a particular request
    pub fn acquire(&mut self, reads: Vec<StateDescriptor>, writes: Vec<StateDescriptor>) {
        self.reads.extend_from_slice(&reads);
        self.writes.extend_from_slice(&writes);

        // TODO - Ensure that necessary state is present

        // TODO - Acquire appropriate resource locks
    }

    /// Release resource locks and compose outgoing update
    pub fn commit(&self) -> StateUpdate {
        let outgoing_update = StateUpdate::compose_from_node(&self.node, &self.writes);

        // TODO - release resource locks

        outgoing_update
    }
}
