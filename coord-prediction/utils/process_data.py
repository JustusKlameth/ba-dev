import pandas as pd
import numpy as np
import json
from geopy import distance
from utils.progress_bar import progress_bar
from utils.calculate_errs import cal_errs_llm
import os
# Assume openai>=1.0.0
from openai import OpenAI
from multiprocessing import Pool


system_message_extract_coords_llama = """Extract any explicit coordinates from the messages you receive.

- **Only** extract coordinates if they are explicitly mentioned in the message (e.g., numerical latitude and longitude).
- **Do not** infer or look up coordinates based on place names, city names, or any other information.
- **If multiple pairs of coordinates are present, extract only the first pair mentioned.**
- If there are no explicit coordinates in the message, reply "nan".

Answer in JSON format only: {"latitude": <e.g., "14.832185">, "longitude": <e.g., "4.212666">}. No explanation, just the answer!"""


class Preprocessor():

    def json(path_input, path_output):
        """
        Verarbeitet Anfragen mithilfe des json-Templates.

        Args:
            path_input (str): Der Pfad der Input-Datei.
            path_output (str): Der Pfad, in dem die Ergebnisse gespeichert werden sollen

        Returns:
            df: Die Ergebnisse.
        """
        # nur Daten verarbeiten, die noch nicht verarbeitet wurden
        if os.path.exists(path_output):
            return Preprocessor.__path_already_exists(path_output)

        df = pd.read_csv(path_input, dtype={'Population': 'Int64'}) # Einlesen
        df = Preprocessor.__extract_coords_json(df) # Koordinaten extrahieren
        df = Preprocessor.__add_error(df) # Fehler berechenen
        df.to_csv(path_output, index=False) # Speichern
        return df
    

    def llm(path_input, path_output, DEEPINFRA_API_KEY, dtype={'Population': 'Int64'}):
        """
        Wertet Antworten eines LLM's aus indem mithilfe eines LLM's Koordinaten extrahiert werden.
        Falls die Datei mit der Auswertung bereits existieren sollte, wird diese genutzt.

        Args:
            path_input (str): Der Pfad der Daten.
            path_output (str): Der Pfad der Auswertung.
            DEEPINFRA_API_KEY (str): API-Key für DeepInfra.
            dtype (dict): Datentype der Daten (Standard: {'Population': 'Int64'}).

        Returns:
            Pandas Dataframe: Die verarbeiteten Daten.
        """
        # zurückgeben der Daten, falls sie bereits vorverarbeitet wurden
        if os.path.exists(path_output):
            return Preprocessor.__path_already_exists(path_output)

        # Laden der Daten
        df = pd.read_csv(path_input, dtype=dtype)
        # Verarbeiten der Daten
        df = Preprocessor.__extract_coords_llama(df, DEEPINFRA_API_KEY) # Koordinaten in json Speichern
        df = Preprocessor.__extract_coords_json(df, row_key='Extract_Coords_Answers') # Koordinaten aus json extrahieren
        df = Preprocessor.__add_error(df) # Fehler hinzufügen
        # Speichern und zurückgeben der Daten
        df.to_csv(path_output, index=False)
        return df
    

    def extract_coords_llama_row(args):
        (model, system_message, index, row, DEEPINFRA_API_KEY) = args

        message = [ { 'role': 'system', 'content': system_message }, { 'role': 'user', 'content': row['Answers'] } ]

        # Create an OpenAI client with your deepinfra token and endpoint
        openai = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
        )

        # Anfrage
        chat_completion = openai.chat.completions.create(
            model=model,
            messages=message,
        )

        answer = chat_completion.choices[0].message.content
        return (index, message, answer)


    def llm_save_parallel(path_input, path_output, DEEPINFRA_API_KEY, dtype={'Population': 'Int64'}, continuous_save=True, continuous_save_n=10, process_number=50):
        # Vorbereitung, falls diese Datei noch nicht existiert (initialisieren der leeren Spalten)
        if not os.path.exists(path_output):
            df = pd.read_csv(path_input, dtype=dtype)
            df['Extract_Coords_Messages'] = None
            df['Extract_Coords_Answers'] = None
            df = df.astype({'Extract_Coords_Messages': 'object', 'Extract_Coords_Answers': 'object'})
            df.to_csv(path_output, index=False)
        
        # Datei laden und noch nicht gemachte Anfragen finden
        df = pd.read_csv(path_output, dtype=dtype)
        df = df.astype({'Extract_Coords_Messages': 'object', 'Extract_Coords_Answers': 'object'})
        indices = df[df[['Extract_Coords_Messages', 'Extract_Coords_Answers']].isnull().all(axis=1)].index.values.tolist()

        # Falls die Anfrage abgeschlossen wurde
        if len(indices) == 0:
            return Preprocessor.__path_already_exists(path_output)

        # Verarbeiten der Daten
        # Prepare tasks for parallel execution
        tasks = []

        for index in indices:
            row = df.iloc[index]
            args = ('meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', system_message_extract_coords_llama, index, row, DEEPINFRA_API_KEY)
            tasks.append(args)

        # Fortschrittsanzeige erstellen und starten
        progress = progress_bar(len(tasks), title='llm-Auswertung')
        progress.start()

        # Use a pool of workers
        # Adjust processes according to your system
        # For large-scale I/O-bound tasks, you might consider a smaller or larger number of workers
        with Pool(processes=process_number) as pool:
            
            # We can use imap_unordered for results as they come in, enabling progress reporting
            for counter, (index, messages, answer) in enumerate(pool.imap_unordered(Preprocessor.extract_coords_llama_row, tasks)):
                
                # Zwischenspeichern
                if continuous_save and counter % continuous_save_n == 0 and counter > 0:
                    df.to_csv(path_output, index=False)

                df.at[index, 'Extract_Coords_Messages'] = messages
                df.at[index, 'Extract_Coords_Answers'] = answer

                progress.update()

        progress.stop()
        df.to_csv(path_output, index=False)

        df = Preprocessor.__extract_coords_json(df, row_key='Extract_Coords_Answers') # Koordinaten aus json extrahieren
        df = Preprocessor.__add_error(df) # Fehler hinzufügen
        # Speichern und zurückgeben der Daten
        df.to_csv(path_output, index=False)
        return df


    def regex(path_input, path_output):
        """
        Verarbeitet Anfragen des original-Templates mit der regex Auswertung aus dem Paper.

        Args:
            path_input (str): Der Pfad der Input-Datei.
            path_output (str): Der Pfad, in dem die Ergebnisse gespeichert werden sollen

        Returns:
            df: Die Ergebnisse.
        """
        # nur Daten verarbeiten, die noch nicht verarbeitet wurden
        if os.path.exists(path_output):
            return Preprocessor.__path_already_exists(path_output)
        
        df = pd.read_csv(path_input, dtype={'Population': 'Int64'}) # Einlesen
        df = cal_errs_llm(df.rename(columns={'Answers': 'output'})) # Umbenennen der Spalten und ausführen der Auswertung mit Regex
        df = df.rename(columns={
            'output': 'Answers',
            'predicted_Latitude': 'LatitudeLLM',
            'predicted_Longitude': 'LongitudeLLM',
            'err': 'Error'
        }) # Spalten zurück benennen
        df = df.astype({'LatitudeLLM': 'float64', 'LongitudeLLM': 'float64', 'Error': 'float64'}) # Datentypen anpassen

        df.to_csv(path_output, index=False)
        return df

    
    def __path_already_exists(path):
        print(f"Achtung! Die Datei {path} existiert bereits.\nDaher werden die zugehörigen Anfragen nicht ausgeführt. Es ist möglich, dass dieses Datei allerdings mit anderen Parametern und/oder Daten erzeugt wurde.")
        return pd.read_csv(path, dtype={'Population': 'Int64'})
    

    def __add_error(df):
        """
        Berechnet bei einem Pandas Dataframe die Distanz zwischen den echten Koordinaten und denen der Antwort des LLM's.
        Diese Distanz wird in der Spalte 'Error' hinzugefügt.
        Falls dabei ein Fehler auftreten sollte, wird 'nan' eingetragen.

        Args:
            df (Pandas Dataframe): Der Input.

        Returns:
            Pandas Dataframe: Der Input mit der neuen Spalte.
        """
        n = df.shape[0]
        error = np.empty(n)

        for index, row in df.iterrows():
            
            try:
                error[index] = distance.distance((row["Latitude"], row["Longitude"]), (row["LatitudeLLM"], row["LongitudeLLM"])).km
            except:
                error[index] = np.nan

        df = df.assign(Error=pd.Series(error).values)
        return df
    

    def __extract_coords_llama(df, DEEPINFRA_API_KEY, system_message=system_message_extract_coords_llama, model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'):
        """
        Extrahiert bei einem Pandas Dataframe aus der Spalte 'Answers' die Koordinaten und speichert sie in der Spalte 'Extract_Coords_Answers' im json-Format.
        Dafür wird ein LLM genutzt (s. Args: model). Die Anfragen an das LLM werden in der Spalte 'Extract_Coords_Messages' gespeichert.

        Args:
            df (Pandas Dataframe): Der Input.
            system_message (str): Die Systemnachricht für das LLM (Standard: s. system_message).
            model (str): Das Modell (Standard: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo).

        Returns:
            Pandas Dataframe: Der Input mit den beiden neuen Spalten.
        """

        # Create an OpenAI client with your deepinfra token and endpoint
        openai = OpenAI(
            api_key=DEEPINFRA_API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
        )

        messages = []
        answers = []

        progress = progress_bar(df.shape[0], title='llm-Auswertung')
        progress.start()

        for _, row in df.iterrows():
            answer = row["Answers"]
            message = [ { 'role': 'system', 'content': system_message }, { 'role': 'user', 'content': answer } ]
            messages.append(message)

            # Anfrage
            chat_completion = openai.chat.completions.create(
                model=model,
                messages=message,
            )

            answers.append(chat_completion.choices[0].message.content)
            progress.update()

        progress.stop()
        df = df.assign(Extract_Coords_Messages=pd.Series(messages).values, Extract_Coords_Answers=pd.Series(answers).values)
        return df
    

    def __extract_coords_json(df, row_key='Answers'):
        """
        Wertet bei einem Pandas Dataframe die Spalte 'Answers' aus indem die Einträge in das 'json'-Format umgewandelt werden.
        Aus diesen Einträgen wird probiert mit den Schlüsseln 'latitude' und 'longitude' die geschätzten Koordinaten zu extrahieren.
        Falls dies funktioniert, werden die Werte in den neuen Spalten 'LatitudeLLM' und 'LongitudeLLM' gespeichert, sonst wird 'nan' eingetragen.
        Dieses neue Dataframe wird zurückgegeben.

        Args:
            df (Pandas Dataframe): Der Input.
            row_key: 'Answers' (str): Die Spalte in der die Koordinaten im json-Format stehen.

        Returns:
            Pandas Dataframe: Der Input mit den beiden neuen Spalten.
        """
        n = df.shape[0]

        latitudes = np.empty(n)
        longitudes = np.empty(n)

        for index, row in df.iterrows():
            answer = row[row_key]

            try:
                coords = json.loads(answer)
                
                latitudes[index] = coords['latitude']
                longitudes[index] = coords['longitude']
            except:
                latitudes[index] = np.nan
                longitudes[index] = np.nan
                
        df = df.assign(LatitudeLLM=pd.Series(latitudes).values, LongitudeLLM=pd.Series(longitudes).values)
        return df
    