class TemplateModel:
    def __init__(self):
        self.data = []

    def add_price(self, value, N_last_max=72, error=2):
        self.data.append(value)
        found, estimation = self.verify_sequence(N_last_max=N_last_max, error=error)
        return estimation if found else None  # oppure ritorna (found, estimation)

    def verify_sequence(self, N_last_max=72, error=2):
        n = len(self.data)

        # Servono almeno N_last_max valori per il match + 1 per la previsione
        if n < N_last_max + 1:
            return False, None

        # Ultima finestra (target) = ultimi N_last_max valori
        # Confronto con finestre passate che abbiano anche un "next" disponibile
        last_start = n - N_last_max

        for step in range(0, last_start):
            # step + N_last_max < n garantito perché step <= last_start - 1
            ok = True
            for k in range(N_last_max):
                if abs(self.data[step + k] - self.data[last_start + k]) > error:
                    ok = False
                    break
            if ok:
                return True, self.data[step + N_last_max]

        return False, None
