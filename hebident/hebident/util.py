import datetime
import os


def nothing_printer(*_, **__):
    pass


class BlankLogger:

    @staticmethod
    def write(msg):
        pass

    def __call__(self, msg):
        pass


class PrintToScreenAndFile:

    def __init__(self, output_filename):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        self.f = open(output_filename, "w")

    def __call__(self, msg):
        self.f.write(msg + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())
        print(msg)


class DurationEstimator:

    def __init__(self, total_objects, context="", window_length=None):
        self.total_objects = total_objects
        self.time_checks = [datetime.datetime.now(), None]
        self.object_checks = [0, None]
        self.context = context
        if self.context:
            self.context += " "
        if window_length is None:
            self.window_length = max(1, int(total_objects/1000))
        else:
            self.window_length = window_length

    def _do_estimate(self, objects_done, now):
        objects_done_in_window = objects_done - self.object_checks[0]
        if objects_done_in_window == 0:
            return "Don't know"
        time_in_window = now - self.time_checks[0]
        time_per_object = time_in_window / objects_done_in_window
        remaining_objects = self.total_objects - objects_done
        remaining_time = time_per_object * remaining_objects
        return now + remaining_time

    def _update_checks(self, objects_done, now):
        if objects_done == 0:
            return
        if objects_done % self.window_length == 0:
            if self.time_checks[1] is None:
                # First time we have self.window_length objects
                # update [1]
                self.time_checks[1] = now
                self.object_checks[1] = objects_done
            elif objects_done > self.object_checks[1]:
                # A later time through - shuffle
                self.time_checks[0] = self.time_checks[1]
                self.object_checks[0] = self.object_checks[1]
                self.time_checks[1] = now
                self.object_checks[1] = objects_done

    def estimate(self, objects_done):
        now = datetime.datetime.now()
        self._update_checks(objects_done, now)
        return self._do_estimate(objects_done, now)

    def print_estimate(self, objects_done, per_call_context=""):
        out = f"{self.context}[{objects_done}/{self.total_objects}] done. Est. finish: {self.estimate(objects_done)}."
        if per_call_context:
            out += f" {per_call_context}."
        print(out)

    def print_estimate_if(self, objects_done, every, per_call_context=""):
        if objects_done % every == 0:
            self.print_estimate(objects_done, per_call_context=per_call_context)


def sort_and_thin_probs(data, number_to_show: int):
    # Data is assumed to be an iterable of two-tuples array with numbers (probabilities) in 2nd entry
    # Sort by highest first
    # But trim to show only 'number_to_show' entries, with rest in 'Other'
    # Useful as a prep step to displaying output from the identifier
    res = []
    else_prob = 0.0
    interim = sorted(data, key=lambda y: y[1], reverse=True)
    for ix, (name, prob) in enumerate(interim):
        if ix < number_to_show:
            res.append((name, "%.1f%%" % (100.0 * prob)))
        else:
            else_prob += prob
    res.append(("Other", "%.1f%%" % (100.0 * else_prob)))
    return res
