"""
Event bus for communication between NormaML components.

This module implements a simple but efficient event bus that enables
loose coupling between system components through publish-subscribe messaging.
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Set, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import weakref


logger = logging.getLogger(__name__)


@dataclass
class Event:
    """
    Base event class for the event bus.
    
    All events in the system inherit from this class and contain
    common metadata about the event occurrence.
    """
    event_type: str
    source: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = self.__class__.__name__


@dataclass
class SystemEvent(Event):
    """Events related to system lifecycle and status."""
    pass


@dataclass
class DataEvent(Event):
    """Events related to data operations and shared memory."""
    pass


@dataclass
class PluginEvent(Event):
    """Events related to plugin lifecycle and operations."""
    pass


@dataclass
class ExecutionEvent(Event):
    """Events related to task and container execution."""
    pass


@dataclass
class ExperimentEvent(Event):
    """Events related to experiment tracking."""
    pass


class EventSubscription:
    """Represents a subscription to events."""
    
    def __init__(self, event_types: Union[str, List[str]], 
                 callback: Callable[[Event], Any],
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 async_callback: bool = False,
                 subscriber_id: Optional[str] = None):
        """
        Initialize event subscription.
        
        Args:
            event_types: Event type(s) to subscribe to
            callback: Function to call when event occurs
            filter_func: Optional filter function for events
            async_callback: Whether callback is async
            subscriber_id: Optional subscriber identifier
        """
        self.subscription_id = str(uuid.uuid4())
        self.event_types = [event_types] if isinstance(event_types, str) else event_types
        self.callback = callback
        self.filter_func = filter_func
        self.async_callback = async_callback
        self.subscriber_id = subscriber_id or "unknown"
        self.created_at = time.time()
        self.call_count = 0
        self.last_called: Optional[float] = None
        self.active = True


class EventBus:
    """
    Simple event bus for component communication.
    
    The EventBus provides:
    - Publish-subscribe messaging between components
    - Event filtering and routing
    - Async and sync event handling
    - Event history and debugging support
    - Thread-safe operations
    """
    
    def __init__(self, max_history: int = 1000, enable_history: bool = True,
                 thread_pool_size: int = 4):
        """
        Initialize the EventBus.
        
        Args:
            max_history: Maximum number of events to keep in history
            enable_history: Whether to maintain event history
            thread_pool_size: Size of thread pool for async processing
        """
        self.max_history = max_history
        self.enable_history = enable_history
        
        # Subscriptions organized by event type
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._all_subscriptions: List[EventSubscription] = []
        
        # Event history
        self._event_history: List[Event] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Async processing
        self._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Statistics
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'total_subscriptions': 0,
            'active_subscriptions': 0
        }
        
        # Weak references to avoid memory leaks
        self._component_refs: Set[weakref.ref] = set()
        
        logger.info(f"EventBus initialized with max_history={max_history}, "
                   f"thread_pool_size={thread_pool_size}")
    
    def subscribe(self, event_types: Union[str, List[str]], 
                  callback: Callable[[Event], Any],
                  filter_func: Optional[Callable[[Event], bool]] = None,
                  async_callback: bool = False,
                  subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to events.
        
        Args:
            event_types: Event type(s) to subscribe to
            callback: Function to call when event occurs
            filter_func: Optional filter function for events
            async_callback: Whether callback is async
            subscriber_id: Optional subscriber identifier
            
        Returns:
            Subscription ID for managing the subscription
        """
        with self._lock:
            subscription = EventSubscription(
                event_types=event_types,
                callback=callback,
                filter_func=filter_func,
                async_callback=async_callback,
                subscriber_id=subscriber_id
            )
            
            # Add to event type mappings
            for event_type in subscription.event_types:
                self._subscriptions[event_type].append(subscription)
            
            # Add to all subscriptions
            self._all_subscriptions.append(subscription)
            
            # Update stats
            self._stats['total_subscriptions'] += 1
            self._stats['active_subscriptions'] += 1
            
            logger.debug(f"Added subscription {subscription.subscription_id} for "
                        f"events {subscription.event_types} from {subscriber_id}")
            
            return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID of subscription to remove
            
        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            # Find and remove subscription
            subscription = None
            for sub in self._all_subscriptions:
                if sub.subscription_id == subscription_id:
                    subscription = sub
                    break
            
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found")
                return False
            
            # Mark as inactive
            subscription.active = False
            
            # Remove from event type mappings
            for event_type in subscription.event_types:
                if event_type in self._subscriptions:
                    self._subscriptions[event_type] = [
                        s for s in self._subscriptions[event_type] 
                        if s.subscription_id != subscription_id
                    ]
            
            # Remove from all subscriptions
            self._all_subscriptions = [
                s for s in self._all_subscriptions 
                if s.subscription_id != subscription_id
            ]
            
            # Update stats
            self._stats['active_subscriptions'] -= 1
            
            logger.debug(f"Removed subscription {subscription_id}")
            return True
    
    def publish(self, event: Event, wait_for_completion: bool = False) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            wait_for_completion: Whether to wait for all callbacks to complete
        """
        with self._lock:
            # Add to history
            if self.enable_history:
                self._event_history.append(event)
                if len(self._event_history) > self.max_history:
                    self._event_history.pop(0)
            
            # Update stats
            self._stats['events_published'] += 1
            
            # Get relevant subscriptions
            subscriptions = self._subscriptions.get(event.event_type, [])
            
            # Also check for wildcard subscriptions (subscribe to "*")
            subscriptions.extend(self._subscriptions.get("*", []))
            
            if not subscriptions:
                logger.debug(f"No subscribers for event type: {event.event_type}")
                return
            
            logger.debug(f"Publishing event {event.event_type} to {len(subscriptions)} subscribers")
            
            # Process subscriptions
            futures = []
            for subscription in subscriptions:
                if not subscription.active:
                    continue
                
                # Apply filter if present
                if subscription.filter_func and not subscription.filter_func(event):
                    continue
                
                # Update subscription stats
                subscription.call_count += 1
                subscription.last_called = time.time()
                
                # Process callback
                try:
                    if subscription.async_callback:
                        # Submit to thread pool for async processing
                        future = self._thread_pool.submit(self._process_async_callback, 
                                                        subscription, event)
                        futures.append(future)
                    else:
                        # Execute synchronously
                        subscription.callback(event)
                    
                    self._stats['events_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing event {event.event_id} "
                               f"in subscription {subscription.subscription_id}: {e}")
            
            # Wait for async callbacks if requested
            if wait_for_completion and futures:
                for future in futures:
                    try:
                        future.result(timeout=30)  # 30 second timeout
                    except Exception as e:
                        logger.error(f"Async callback failed: {e}")
    
    def _process_async_callback(self, subscription: EventSubscription, event: Event) -> None:
        """Process an async callback in the thread pool."""
        try:
            if asyncio.iscoroutinefunction(subscription.callback):
                # Run async callback
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(subscription.callback(event))
                finally:
                    loop.close()
            else:
                # Run sync callback in thread
                subscription.callback(event)
        except Exception as e:
            logger.error(f"Error in async callback: {e}")
    
    def publish_system_event(self, event_type: str, source: str, 
                           data: Optional[Dict[str, Any]] = None) -> None:
        """
        Convenience method for publishing system events.
        
        Args:
            event_type: Type of system event
            source: Source component name
            data: Optional event data
        """
        event = SystemEvent(
            event_type=event_type,
            source=source,
            data=data or {}
        )
        self.publish(event)
    
    def publish_data_event(self, event_type: str, source: str,
                          data: Optional[Dict[str, Any]] = None) -> None:
        """
        Convenience method for publishing data events.
        
        Args:
            event_type: Type of data event
            source: Source component name
            data: Optional event data
        """
        event = DataEvent(
            event_type=event_type,
            source=source,
            data=data or {}
        )
        self.publish(event)
    
    def publish_plugin_event(self, event_type: str, source: str,
                           data: Optional[Dict[str, Any]] = None) -> None:
        """
        Convenience method for publishing plugin events.
        
        Args:
            event_type: Type of plugin event
            source: Source component name
            data: Optional event data
        """
        event = PluginEvent(
            event_type=event_type,
            source=source,
            data=data or {}
        )
        self.publish(event)
    
    def get_event_history(self, event_type: Optional[str] = None,
                         source: Optional[str] = None,
                         limit: Optional[int] = None) -> List[Event]:
        """
        Get event history with optional filtering.
        
        Args:
            event_type: Filter by event type
            source: Filter by event source
            limit: Maximum number of events to return
            
        Returns:
            List of events matching criteria
        """
        with self._lock:
            events = self._event_history[:]
            
            # Apply filters
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if source:
                events = [e for e in events if e.source == source]
            
            # Apply limit
            if limit:
                events = events[-limit:]
            
            return events
    
    def get_subscription_info(self) -> Dict[str, Any]:
        """
        Get information about current subscriptions.
        
        Returns:
            Dictionary with subscription statistics
        """
        with self._lock:
            active_by_type = defaultdict(int)
            for event_type, subs in self._subscriptions.items():
                active_by_type[event_type] = len([s for s in subs if s.active])
            
            return {
                'total_subscriptions': len(self._all_subscriptions),
                'active_subscriptions': len([s for s in self._all_subscriptions if s.active]),
                'subscriptions_by_type': dict(active_by_type),
                'statistics': self._stats.copy()
            }
    
    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()
            logger.debug("Event history cleared")
    
    def register_component(self, component: Any) -> None:
        """
        Register a component with the event bus.
        
        This creates a weak reference to avoid memory leaks.
        
        Args:
            component: Component to register
        """
        try:
            ref = weakref.ref(component)
            self._component_refs.add(ref)
            logger.debug(f"Registered component: {component.__class__.__name__}")
        except TypeError:
            # Object is not weakly referenceable
            logger.debug(f"Could not create weak reference for component: {type(component)}")
    
    def cleanup_dead_references(self) -> None:
        """Clean up dead weak references."""
        with self._lock:
            # Remove dead references
            dead_refs = {ref for ref in self._component_refs if ref() is None}
            self._component_refs -= dead_refs
            
            if dead_refs:
                logger.debug(f"Cleaned up {len(dead_refs)} dead component references")
    
    def shutdown(self) -> None:
        """Shutdown the event bus and cleanup resources."""
        logger.info("Shutting down EventBus")
        
        with self._lock:
            # Clear all subscriptions
            self._subscriptions.clear()
            self._all_subscriptions.clear()
            
            # Clear history
            if self.enable_history:
                self._event_history.clear()
            
            # Clean up component references
            self._component_refs.clear()
            
            # Reset stats
            self._stats = {
                'events_published': 0,
                'events_processed': 0,
                'total_subscriptions': 0,
                'active_subscriptions': 0
            }
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("EventBus shutdown completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore cleanup errors during destruction


# Global event bus instance (optional convenience)
_global_event_bus: Optional[EventBus] = None


def get_global_event_bus() -> EventBus:
    """
    Get or create the global event bus instance.
    
    Returns:
        Global EventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_global_event_bus(event_bus: EventBus) -> None:
    """
    Set the global event bus instance.
    
    Args:
        event_bus: EventBus instance to set as global
    """
    global _global_event_bus
    _global_event_bus = event_bus