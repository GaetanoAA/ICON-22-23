from pyswip import Prolog


class PrologReasoner:
    def __init__(self, kb_file_path, threshold=0):
        self.prolog = Prolog()
        self.prolog.consult(kb_file_path)
        self.threshold = threshold

    def perform_reasoning(self):
        query_count_parasitized = "count_parasitized(N)"
        query_count_uninfected = "count_uninfected(N)"

        count_parasitized = list(self.prolog.query(query_count_parasitized))[0]["N"]
        count_uninfected = list(self.prolog.query(query_count_uninfected))[0]["N"]

        query_above_threshold = "immagine(X, _, L), L = [_ | Rest], last(Rest, B), B > 160.0"
        query_below_threshold = "immagine(X, _, L), L = [_ | Rest], last(Rest, B), B < 160.0"

        above_threshold_images = list(self.prolog.query(query_above_threshold))
        below_threshold_images = list(self.prolog.query(query_below_threshold))

        return count_parasitized, count_uninfected, above_threshold_images, below_threshold_images

