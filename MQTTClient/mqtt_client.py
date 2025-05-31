# mqtt_client.py

import paho.mqtt.client as mqtt
import threading
import time

from Logger import Logger


class MQTTClient:
    def __init__(self, broker='localhost', port=1883, keepalive=60):
        """
        Initializes the MQTT client with broker configuration.
        """
        self.broker = broker
        self.port = port
        self.keepalive = keepalive
        self.username = None
        self.password = None

        self.client = mqtt.Client()
        self.callbacks = {}  # Maps topics to callback functions

        # Register internal event handlers
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self._connect_lock = threading.Lock()
        self._running = False

    def set_config(self, broker=None, port=None, username=None, password=None):
        Logger.debug(f"Setting MQTT config: broker={broker}, port={port}, username={username}")
        """
        Updates the MQTT connection configuration.
        Can only be safely applied before start(), or it will reconnect using new settings.
        """
        with self._connect_lock:
            # Update config
            if broker:
                self.broker = broker
            if port:
                # force port to int
                self.port = int(port)
            if username:
                self.username = username
            if password:
                self.password = password

            # Reconfigure client if running
            if self._running:
                self.client.loop_stop()
                self.client = mqtt.Client()
                self.client.on_connect = self._on_connect
                self.client.on_disconnect = self._on_disconnect
                self.client.on_message = self._on_message

                if self.username and self.password:
                    self.client.username_pw_set(self.username, self.password)

                self.client.loop_start()
                self._connect()

    def start(self):
        """
        Starts the MQTT client loop and attempts initial connection.
        """
        with self._connect_lock:
            if not self._running:
                self._running = True
                if self.username and self.password:
                    self.client.username_pw_set(self.username, self.password)
                self.client.loop_start()
                self._connect()

    def stop(self):
        """
        Stops the MQTT client loop and disconnects from the broker.
        """
        with self._connect_lock:
            self._running = False
            self.client.loop_stop()
            self.client.disconnect()

    def _connect(self):
        """
        Internal method to connect to the MQTT broker with retry logic.
        """
        while self._running:
            try:
                self.client.connect(self.broker, self.port, self.keepalive)
                break
            except Exception as e:
                Logger.error(f"[MQTT] Connection failed: {e}. Retrying in 5 seconds.")
                time.sleep(5)

    def _on_connect(self, client, userdata, flags, rc):
        """
        Internal callback when the client connects to the broker.
        Re-subscribes to all previously subscribed topics.
        """
        if rc == 0:
            Logger.info("[MQTT] Connected to broker.")
            for topic in self.callbacks:
                client.subscribe(topic)
        else:
            Logger.critical(f"[MQTT] Failed to connect, return code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """
        Internal callback when the client disconnects.
        Automatically attempts to reconnect.
        """
        if self._running:
            Logger.warning("[MQTT] Disconnected from broker. Attempting to reconnect...")
            self._connect()

    def _on_message(self, client, userdata, msg):
        """
        Internal callback for incoming messages.
        Routes messages to the appropriate registered callback.
        """
        callback = self.callbacks.get(msg.topic)
        if callback:
            try:
                callback(msg.topic, msg.payload.decode())
            except Exception as e:
                Logger.error(f"[MQTT] Error in callback for topic '{msg.topic}': {e}")

    def publish(self, topic, payload, qos=0, retain=False):
        """
        Publishes a message to the specified topic.
        """
        self.client.publish(topic, payload, qos, retain)

    def subscribe(self, topic, callback):
        """
        Subscribes to a topic and registers a callback to be invoked when a message is received.
        """
        self.callbacks[topic] = callback
        self.client.subscribe(topic)

    def wait_for_broker_ready(self, timeout=30, interval=0.5):
        """
        Blocks until the client is connected to the broker or timeout is reached.

        :param timeout: Max number of seconds to wait
        :param interval: Time between checks in seconds
        :return: True if connected, False if timeout occurred
        """
        Logger.info("[MQTT] Waiting for broker to be ready...")
        waited = 0
        while waited < timeout:
            if self.client.is_connected():
                Logger.info("[MQTT] Broker is ready.")
                return True
            time.sleep(interval)
            waited += interval

        Logger.error(f"[MQTT] Timeout waiting for broker to be ready after {timeout} seconds.")
        return False


# Global shared instance
mqtt_client = MQTTClient()