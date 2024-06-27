#!/usr/bin/env python3
import time
import logging

from datetime import datetime
from nano_llm import Plugin, StopTokens


class EventFilter(Plugin):
    """
    Plugin for matching incoming text
    """
    def __init__(self, filters: str=None, tags: str=None, server=None, **kwargs):
        """
        Filter incoming text and trigger events when search conditions are met.
        
        Args:
          filters (str or list[str]): Comma-separated list of search strings (use + for AND instead of OR)
          tags (str): Label or description to tag the events with (this gets used in notifications)
        """
        super().__init__(outputs=['events', 'alerts'], **kwargs)

        self.tags = tags
        self.prompt = None
        self.history = []
        self.server = server
        self.filter_type = 'or'
        
        self.add_parameter('tags', default=tags)
        self.add_parameter('filters', default=filters)

        if self.server:
            self.server.add_message_handler(self.on_websocket)
        
    def process(self, text, prompt=None, **kwargs):
        """
        Detect if the criteria for an event filters occurred in the incoming text.
        
        Args:
          input (str): text to filter/search against
                                          
        Returns:
          Event dict if a new event occurred, false otherwise
        """
        filters = self.filter_text(text, self.filter_list, op=self.filter_type)
        
        if not text or not filters:
            if self.history and 'end' not in self.history[-1]:
                self.on_event_end(self.history[-1])
            return
            
        new_event = False
        
        if not self.history:
            new_event = True
        elif 'end' in self.history[-1]:
            new_event = True
        elif self.history[-1]['filters'] != filters:
            new_event = True
            self.on_event_end(self.history[-1])
            
        if new_event:
            event = self.on_event_begin(text, filters, prompt=prompt)
            self.output(event['tags'], channel=1, final=True)
            return event
        else:
            self.history[-1]['last'] = time.time()
            self.send_events(self.history)
  
    @property
    def filters(self):
        return self._filters
        
    @filters.setter
    def filters(self, filters):
        if not filters:
            self.filter_list = None
            self._filters = None
            return
         
        self._filters = filters
        filters = filters.split('+')
        
        if len(filters) > 1:
            self.filter_type = 'and'
        else:
            filters = filters[0].split(',')
            self.filter_type = 'or'
            
        self.filter_list = [x.strip().lower() for x in filters]
 
    def filter_text(self, text, filters, op='or'):
        if not text:
            return []
        if not filters:
            return []
        
        matches = [x for x in filters if x in text.lower()]
        
        if op == 'and' and len(matches) != len(filters):
            return []
            
        return matches
                
    def on_event_begin(self, text, filters, prompt=None):
        event = {
            'id': len(self.history),
            'text': text.strip(),
            'filters': filters,
            'begin': time.time(),
            'last': time.time(),
        }
        
        if prompt:
            event['prompt'] = prompt
        
        if self.tags:
            event['tags'] = self.tags if self.tags else self.filters
            alert_text = f"EVENT OCCURRED  '{event['tags']}'"
        else:
            alert_text = f"EVENT OCCURRED  {event['filters']}"

        event['alert'] = self.send_alert(alert_text, category='event_begin', level='warning')
        
        self.history.append(event)
        self.send_events(self.history)
        
        return event
        
    def on_event_end(self, event):
        event['end'] = time.time()
        #self.server.send_message({'end_alert': event['alert']['id']})
        alert_text = f"EVENT FINISHED  '{event.get('tags', event['filters'])}'  (duration {event['end']-event['begin']:.1f} seconds)"
        self.send_alert(alert_text, category='event_end', level='success')
        self.send_events(self.history)

    def format_event(self, event):
        event = event.copy()
        time_format = '%-I:%M:%S'
        
        event['begin'] = datetime.fromtimestamp(event['begin']).strftime(time_format)
        event['last'] = datetime.fromtimestamp(event['last']).strftime(time_format)
        
        if 'end' in event:
            event['end'] = datetime.fromtimestamp(event['end']).strftime(time_format)
        else:
            event['end'] = event['last']
            
        event['filters'] = str(event['filters'])
        
        for stop in StopTokens:
            event['text'] = event['text'].replace(stop, '')
            
        del event['alert']
        return event
        
    def send_events(self, events, max_events=10):
        if self.server is None:
            return
        if max_events and len(events) > max_events:
            events = events[-max_events:]
        events = [self.format_event(event) for event in events]
        self.server.send_message({'events': events})
      
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        if not isinstance(msg, dict):  # msg_type != WebServer.MESSAGE_JSON:
            return 
        if 'event_filters' in msg:
            self.filters = msg['event_filters']
            logging.info(f'set event filters to "{msg["event_filters"]}" {self.filters}')
        elif 'event_tags' in msg:
            self.tags = msg['event_tags']
                
