import json
import threading
import time


class ProgressStore:
    def __init__(self, with_results=False, with_counter=False):
        self._events = {}
        self._results = {}
        self._lock = threading.Lock()
        self._with_results = with_results
        self._with_counter = with_counter
        self._counter = 0

    def next_id(self):
        if not self._with_counter:
            raise RuntimeError("This store does not generate ids")
        with self._lock:
            self._counter += 1
            return self._counter

    def update(self, key, message, progress, done=False, error=None, result=None):
        entry = {'message': message, 'progress': progress, 'done': done}
        if error:
            entry['error'] = error
        with self._lock:
            self._events.setdefault(key, []).append(entry)
            if self._with_results and result is not None:
                self._results[key] = result

    def events_since(self, key, cursor=0):
        with self._lock:
            events = self._events.get(key, [])
            return events[cursor:], len(events)

    def get_result(self, key):
        with self._lock:
            return self._results.get(key)


def stream_events(store, key, interval_seconds=0.5, include_result_on_done=False):
    cursor = 0
    while True:
        events, new_cursor = store.events_since(key, cursor)
        for event in events:
            if include_result_on_done and event.get('done'):
                payload = dict(event)
                result = store.get_result(key)
                if result is not None:
                    payload['result'] = result
                yield f"data: {json.dumps(payload)}\n\n"
                return
            yield f"data: {json.dumps(event)}\n\n"
            if event.get('done') or event.get('error'):
                return
        cursor = new_cursor
        time.sleep(interval_seconds)
