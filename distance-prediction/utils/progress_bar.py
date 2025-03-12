import time

class progress_bar:
    """
    Zeigt eine Fortschrittsanzeige an.
    """
    n = None
    counter = None
    start_time = None
    bar_length = None


    def __init__(self, n, bar_length=50, title=None):
        """
        Setzt die Gesamtanzahl der Schritte und die Länge der Fortschittsanzeige.

        Args:
            n (int): Die Gesamtanzahl der Schritte.
            bar_length: 50 (int): Die Länge der Anzeige.
            title: None (str): Überschrift.
        """
        self.n = n
        self.bar_length = bar_length
        self.title = title

        self.counter = 0


    def start(self):
        """
        Startet die Anzeige und sie wird mit dem Fortschritt 0% angezeigt.
        """
        if self.title != None:
            print(self.title + ":")
        
        self.start_time = time.time()
        self.update(step=0)


    def stop(self):
        """
        Beendet die Anzeige und sie wird dauerhaft mit dem Fortschritt 100% angezeigt.
        """
        duration = time.time() - self.start_time
        bar = '\u2593' * self.bar_length

        print(f"\rZeit: {duration:5.1f}s, T-{0:<5.1f}s, {bar} {100:3.0f}%")


    def update(self, step=1):
        """
        Updatet die Anzeige und sie wird mit dem neuen Fortschritt angezeigt.
        
        Args:
            step: 1 (int): Die Anzahl der Schritte, die in der zwischenzeit gemacht wurden.
        """
        self.counter += step

        percent = self.counter / self.n

        filled_length = int(self.bar_length * percent)
        bar = '\u2593' * filled_length + '\u2591' * (self.bar_length - filled_length)

        duration = time.time() - self.start_time

        remaining_seconds = float('nan')

        if self.counter != 0:
            remaining_seconds = (duration / self.counter) * (self.n - self.counter)

        print(f"\rZeit: {duration:5.1f}s, T-{remaining_seconds:<5.1f}s, {bar} {100 * percent:3.0f}%", end="", flush=True)


    def get_progress_bar_string(percent, bar_length=50):
        """
        Gibt eine Zeichenkette zurück, die einen Fortschrittsbalken darstellt.

        Args:
            percent (float): Der Fortschritt in Prozent (0-1).
            bar_length (int, optional): Die Länge des Fortschrittsbalkens. Standardmäßig 50.

        Returns:
            str: Die Zeichenkette, die den Fortschrittsbalken darstellt.
        """
        filled_length = int(bar_length * percent)
        bar = '\u2593' * filled_length + '\u2591' * (bar_length - filled_length)
        return bar


    def print_progress(start_time, counter, n):
        """
        Gibt den Fortschritt der Verarbeitung aus.

        Args:
            start_time (float): Die Startzeit der Verarbeitung.
            counter (int): Die Anzahl der bisherigen Schritte.
            n (int): Die Anzahl aller Schritte.
        """
        duration = time.time() - start_time
        percent = counter / n

        print(f"\rZeit: {duration:5.1f}s, T-{(n - counter) / (counter / duration):<5.1f}s, {progress_bar.get_progress_bar_string(percent)} {100 * percent:3.0f}%", end="", flush=True)