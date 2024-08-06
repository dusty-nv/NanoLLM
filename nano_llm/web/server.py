#!/usr/bin/env python3
import os
import io
import PIL
import ssl
import json
import time
import flask
import queue
import struct
import pprint
import logging
import datetime
import threading
import traceback

from nano_llm.utils import ArgParser

from websockets.sync.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed

class WebServer():
    MESSAGE_JSON = 0     #: JSON websocket message (dict)
    MESSAGE_TEXT = 1     #: Text websocket message (str)
    MESSAGE_BINARY = 2   #: Binary websocket message (bytes)
    MESSAGE_FILE = 3     #: File upload from client (bytes)
    MESSAGE_AUDIO = 4    #: Audio samples (bytes, int16)
    MESSAGE_IMAGE = 5    #: Image message (PIL.Image)

    Instance = None      #: Singleton instance
    MessageHandlers = [] #: Message handlers
    
    def __init__(self, web_host='0.0.0.0', web_port=8050, ws_port=49000,
                 ssl_cert=None, ssl_key=None, root=None, index='index.html',
                 mounts={'/tmp/uploads':'/uploads'}, msg_callback=None, web_trace=False,
                 **kwargs):
        """
        Create HTTP/HTTPS Flask webserver with websocket messaging.
    
        Use this by either creating an instance and providing ``msg_callback``,
        or inherit from it and implement ``on_message()`` in a subclass.
        You can also add Flask routes to Webserver.app before ``start()`` is called.
    
        Args:
        
          web_host (str): network interface to bind to (0.0.0.0 for all)
          web_port (int): port to serve HTTP/HTTPS webpages on
          ws_port (int): port to use for websocket communication
          ssl_cert (str): path to PEM-encoded SSL/TLS cert file for enabling HTTPS
          ssl_key (str): path to PEM-encoded SSL/TLS cert key for enabling HTTPS
          root (str): the root directory for serving site files (should have static/ and template/)
          index (str): the name of the site's index page (should be under web/templates)
          upload_dir (str): the path to save files uploaded from client (or None to disable uploads)
          msg_callback (callable): websocket message handler (see WebServer.on_message() for signature)
          web_trace (bool): if true, additional debug messages will be printed when --log-level=debug
          
        The kwargs are passed as variables to the Jinja render_template() used in the index file.
        """
        WebServer.Instance = self
        
        self.host = web_host
        self.port = web_port
        self.root = root
        
        self.trace = web_trace
        self.index = index
        self.kwargs = kwargs
        self.mounts = mounts
        self.upload_dir = None
        self.alert_count = 0
        self.websocket = None
        
        if not self.root:
            self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../web'))
            
        self.msg_count_rx = 0
        self.msg_count_tx = 0

        self.add_message_handler(msg_callback)
            
        # flask server
        self.app = flask.Flask(__name__, 
            static_folder=os.path.join(self.root, 'static'),
            template_folder=os.path.join(self.root, 'templates')
        )
        
        self.app.use_x_sendfile = True
        
        # setup default index route
        self.app.add_url_rule('/', view_func=self.send_index, methods=['GET'])
        
        # setup mounted paths
        for path, mount in self.mounts.items():
            if path.startswith('/tmp'):
                os.makedirs(path, exist_ok=True)
            if 'upload' in path or 'upload' in mount:
                self.upload_dir = path
            logging.info(f"mounting webserver path {path} to {mount}")
            self.app.add_url_rule(f"{mount}/<path:path>", view_func=SendFromDirectory(path).send, endpoint=path, methods=['GET'])
            
        logging.debug(f"webserver root directory: {self.root}   upload directory: {self.upload_dir}")
                 
        # SSL / HTTPS
        self.ssl_key = ssl_key
        self.ssl_cert = ssl_cert
        self.ssl_context = None
        self.web_protocol = "http"
        
        if self.ssl_cert and self.ssl_key:
            self.web_protocol = "https"
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=self.ssl_cert, keyfile=self.ssl_key)
            
        # websocket
        self.ws_port = ws_port
        self.kwargs['ws_port'] = ws_port

        self.ws_server = websocket_serve(self.on_websocket, host=self.host, port=self.ws_port, ssl_context=self.ssl_context, max_size=None, compression=None)
        self.ws_thread = threading.Thread(target=lambda: self.ws_server.serve_forever(), daemon=True)
        self.web_thread = threading.Thread(target=lambda: self.app.run(host=self.host, port=self.port, ssl_context=self.ssl_context, debug=True, use_reloader=False), daemon=True)
        
        # https://stackoverflow.com/a/52282788
        logging.getLogger('asyncio').setLevel(logging.INFO)
        logging.getLogger('asyncio.coroutines').setLevel(logging.INFO)
        logging.getLogger('websockets.server').setLevel(logging.INFO)
        logging.getLogger('websockets.protocol').setLevel(logging.INFO)

    def start(self):
        """
        Call this to start the webserver listening for new connections.
        It will start new worker threads and then return control to the user.
        """
        logging.info(f"starting webserver @ {self.web_protocol}://{self.host}:{self.port}")
        
        self.ws_thread.start()
        self.web_thread.start()

    @property
    def connected(self):
        """
        Returns true if the server is connected to any clients, otherwise false.
        """
        return (self.num_clients > 0)
        
    @property
    def num_clients(self):
        """
        Returns the number of actively connected clients.
        """
        return 0 if self.websocket is None else 1
        
    @classmethod 
    def add_listener(cls, callback):
        """
        Register a message handler that will be called when new websocket messages are recieved.
        """
        cls.add_message_handler(callback)
        
    @classmethod 
    def add_message_handler(cls, callback):
        """
        Register a message handler that will be called when new websocket messages are recieved.
        """
        if callback is None:
            return
            
        if not isinstance(callback, list):
            callback = [callback]
            
        cls.MessageHandlers += callback
            
    def on_message(self, payload, payload_size=None, msg_type=MESSAGE_JSON, msg_id=None, metadata=None, timestamp=None, path=None, **kwargs):
        """
        Handler for recieved websocket messages. Implement this in a subclass to process messages,
        otherwise ``msg_callback`` needs to be provided during initialization.
        
        Args:
        
          payload (dict|str|bytes): If this is a JSON message, will be a dict.
                                    If this is a text message, will be a string.
                                    If this is a binary message, will be a bytes array.  
                                      
          payload_size (int): size of the payload (in bytes)              
          msg_type (int): MESSAGE_JSON (0), MESSAGE_TEXT (1), MESSAGE_BINARY (2)
          msg_id (int): the monotonically-increasing message ID number
          metadata (str): message-specific string or other data
          timestamp (int): time that the message was sent
          path (str): if this is a file or image upload, the file path on the server
        """
        if self.MessageHandlers:
            for callback in WebServer.MessageHandlers:
                try:
                    callback(payload, payload_size=payload_size, msg_type=msg_type, msg_id=msg_id, 
                             metadata=metadata, timestamp=timestamp, path=path, **kwargs)
                except Exception as error:
                    logging.error(f"Exception occurred handling websocket message:\n\n{pprint.pformat(payload, indent=2) if msg_type==WebServer.MESSAGE_JSON else ''}\n{traceback.format_exc()}")
        else:
            raise NotImplementedError(f"{type(self)} did not implement on_message or have a msg_callback provided")
     
    def send_message(self, payload, type=None, timestamp=None):
        """
        Send a websocket message to client.
        """
        if timestamp is None:
            timestamp = time.time() * 1000
         
        encoding = None
        
        if type is None:
            if isinstance(payload, str):
                type = WebServer.MESSAGE_TEXT
                encoding = 'utf-8'
            elif isinstance(payload, bytes):
                type = WebServer.MESSAGE_BINARY
            else:
                type = WebServer.MESSAGE_JSON
                encoding = 'ascii'
        
        if self.websocket is None:
            logging.debug(f"send_message() - no websocket clients connected, dropping {self.msg_type_str(type)} message")
            return
            
        if self.trace and logging.getLogger().isEnabledFor(logging.DEBUG):
            msg_text = '\n' + pprint.pformat(payload) if type <= WebServer.MESSAGE_TEXT else ''
            logging.debug(f"sending {WebServer.msg_type_str(type)} websocket message (type={type} size={len(payload)}){msg_text}")
                    
        if type == WebServer.MESSAGE_JSON and not isinstance(payload, str):  # json.dumps() might have already been called
            #print('sending JSON', payload)
            payload = json.dumps(payload)
            
        if not isinstance(payload, bytes):
            if encoding is not None:
                payload = bytes(payload, encoding=encoding)
            else:
                payload = bytes(payload)
                
        # do we even need this queue at all and can the websocket just send straight away?
        try:
            self.websocket.send(b''.join([
                #
                # 32-byte message header format:
                #
                #   0   uint64  message_id    (message_count_tx)
                #   8   uint64  timestamp     (milliseconds since Unix epoch)
                #   16  uint16  magic_number  (42)
                #   18  uint16  message_type  (0=json, 1=text, >=2 binary)
                #   20  uint32  payload_size  (in bytes)
                #   24  uint32  unused        (padding)
                #   28  uint32  unused        (padding)
                #
                struct.pack('!QQHHIII',
                    self.msg_count_tx,
                    int(timestamp),
                    42, type,
                    len(payload),
                    0, 0,
                ),
                payload
            ]))
            self.msg_count_tx += 1
        except Exception as err:
            logging.warning(f"failed to send websocket message to client ({err})")
    
    def send_alert(self, message, level='warning', category='', timeout=3.5):
        alert = {
            'id': self.alert_count,
            'time': datetime.datetime.now().strftime('%-I:%M:%S'),
            'message': message,
            'level': level,
            'category': category,
            'timeout': int(timeout*1000),
        }
        
        self.send_message({'alert': alert})
        self.alert_count = self.alert_count + 1
        
        if level == 'error':
            logging.error(message)
        elif level == 'warning':
            logging.warning(message)
        else:
            logging.info(message)
 
        return alert
        
    def on_websocket(self, websocket):      
        self.websocket = websocket  # TODO handle multiple clients
        remote_address = websocket.remote_address

        logging.info(f"new websocket connection from {remote_address}")

        '''
        # empty the queue from before the connection was made
        # (otherwise client will be flooded with old messages)
        # TODO implement self.connected so the ws_queue doesn't grow so large without webclient connected...
        while True:
            try:
                self.ws_queue.get(block=False)
            except queue.Empty:
                break
        '''
        
        if self.MessageHandlers:
            for callback in WebServer.MessageHandlers:
                try:
                    callback({'client_state': 'connected'}, msg_type=WebServer.MESSAGE_JSON, timestamp=int(time.time()*1000))
                except Exception as error:
                    logging.error(f"Exception occurred handling client_state 'connected' message\n{traceback.format_exc()}")
     
        #listener_thread = threading.Thread(target=self.websocket_listener, args=[websocket], daemon=True)
        #listener_thread.start()

        try:
            self.websocket_listener(websocket)
        except ConnectionClosed as closed:
            logging.info(f"websocket connection with {remote_address} was closed")
            if self.websocket == websocket: # if the client refreshed, the new websocket may already be created
                self.websocket = None
              
        '''
        while True:
            try:
                websocket.send(self.ws_queue.get())
            except ConnectionClosed as closed:
                logging.info(f"websocket connection with {remote_address} was closed")
                return
        '''
        
    def websocket_listener(self, websocket):
        logging.info(f"listening on websocket connection from {websocket.remote_address}")

        header_size = 32
            
        while True:
            msg = websocket.recv()
            
            if isinstance(msg, str):
                logging.warning(f'dropping text-mode websocket message from {websocket.remote_address} "{msg}"')
                continue
                
            if len(msg) <= header_size:
                logging.warning(f"dropping invalid websocket message from {websocket.remote_address} (size={len(msg)})")
                continue
                
            msg_id, timestamp, magic_number, msg_type, payload_size = \
                struct.unpack_from('!QQHHI', msg)
            
            metadata = msg[24:32].split(b'\x00')[0].decode()
            
            if magic_number != 42:
                logging.warning(f"dropping invalid websocket message from {websocket.remote_address} (magic_number={magic_number} size={len(msg)})")
                continue

            if msg_id != self.msg_count_rx:
                logging.debug(f"recieved websocket message from {websocket.remote_address} with out-of-order ID {msg_id}  (last={self.msg_count_rx})")
                self.msg_count_rx = msg_id
                
            self.msg_count_rx += 1
            msgPayloadSize = len(msg) - header_size
            
            if payload_size != msgPayloadSize:
                logging.warning(f"recieved invalid websocket message from {websocket.remote_address} (payload_size={payload_size} actual={msgPayloadSize}");
            
            payload = msg[header_size:]
            
            if msg_type == WebServer.MESSAGE_JSON:  # json
                payload = json.loads(payload)
            elif msg_type == WebServer.MESSAGE_TEXT:  # text
                payload = payload.decode('utf-8')

            if self.trace and msg_type != WebServer.MESSAGE_AUDIO and logging.getLogger().isEnabledFor(logging.DEBUG):
                msg_text = '\n' + pprint.pformat(payload) if msg_type <= WebServer.MESSAGE_TEXT else ''
                logging.debug(f"recieved {WebServer.msg_type_str(msg_type)} websocket message from {websocket.remote_address} (type={msg_type} size={payload_size}){msg_text}")
                
            # save uploaded files/images to the upload dir
            filename = None
                
            if self.upload_dir and metadata and (msg_type == WebServer.MESSAGE_FILE or msg_type == WebServer.MESSAGE_IMAGE):
                filename = f"{datetime.datetime.utcfromtimestamp(timestamp/1000).strftime('%Y%m%d_%H%M%S')}.{metadata}"
                filename = os.path.join(self.upload_dir, filename)
                threading.Thread(target=self.save_upload, args=[payload, filename]).start()
             
            # decode images in-memory
            if msg_type == WebServer.MESSAGE_IMAGE:
                try:
                    payload = PIL.Image.open(io.BytesIO(payload))
                    if filename:
                        payload.filename = filename
                except Exception as err:
                    print(err)
                    logging.error(f"failed to load invalid/corrupted {metadata} image uploaded from client")
                    
            self.on_message(payload, payload_size=payload_size, msg_type=msg_type, msg_id=msg_id, metadata=metadata, timestamp=timestamp, path=filename)
  
    def save_upload(self, payload, path):
        logging.debug(f"saving client upload to {path}")
        with open(path, 'wb') as file:
            file.write(payload)

    def send_index(self):
        return flask.render_template(self.index, **self.kwargs)
    
    @staticmethod
    def msg_type_str(type):
        if type == WebServer.MESSAGE_JSON:
            return "json"
        elif type == WebServer.MESSAGE_TEXT:
            return "text"
        elif type == WebServer.MESSAGE_BINARY:
            return "binary"
        elif type == WebServer.MESSAGE_FILE:
            return "file"
        elif type == WebServer.MESSAGE_AUDIO:
            return "audio"
        elif type == WebServer.MESSAGE_IMAGE:
            return "image"
        else:
            raise ValueError(f"unknown message type {type}")
            

class SendFromDirectory():
    def __init__(self, root):
        self.root = root
    
    def send(self, path):
        return flask.send_from_directory(self.root, path, conditional=False, max_age=120, use_x_sendfile=True)
        
        
if __name__ == "__main__":
    parser = ArgParser(extras=['web', 'log'])
    
    parser.add_argument("--index", "--page", type=str, default="index.html", help="the filename of the site's index html page (should be under web/templates)") 
    parser.add_argument("--root", type=str, default=None, help="the root directory for serving site files (should have static/ and template/")
    
    args = parser.parse_args()

    webserver = WebServer(**vars(args))
        
    webserver.start()
    webserver.web_thread.join()
    
