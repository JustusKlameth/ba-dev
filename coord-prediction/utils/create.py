import os
import json
import copy

import pandas as pd
import numpy as np

import pycountry
from openai import OpenAI
from utils.progress_bar import progress_bar

from multiprocessing import Pool

dtype = dict(zip(['Country', 'City', 'Population', 'Latitude', 'Longitude'], ['object', 'object', 'Int64', 'float64', 'float64']))
columns = ['Country', 'City', 'Population', 'Latitude', 'Longitude']


def create_samples(
        samples_path,
        data_path='../data/worldcitiespop.csv',
        columns=columns,
        dtype=dtype,
        only_big_cities=True,
        n=1000
    ):
    """
    Erstellt einen neuen Datensatz mit Städten.

    Args:
        samples_path (str): Pfad der Datei, in der der neue Datensatz gespeichert werden soll.
        data_path: '../data/worldcitiespop.csv' (str): Pfad des vollständigen Datensatzes.
        columns: ['Country', 'City', 'Population', 'Latitude', 'Longitude'] (['str']): Spalten, die gespeichert werden sollen.
        dtype: dict(zip(['Country', 'City', 'Population', 'Latitude', 'Longitude'], ['object', 'object', 'Int64', 'float64', 'float64'])): Typen der Spalten.
        only_big_cities: True (bool): Sollen nur Städte mit min. 100000 Einwohnern gespeichert werden?
        n: 1000 (int): Anzahl der Städte. (für None, alle Daten)

    Returns:
        True: wenn ein neuer Datensatz erstellt wurde
        False: falls bereits eine Datei mit dem Pfad existiert
    """
    if os.path.exists(samples_path): # wenn die Daten noch nicht existieren:
        return False

    # Daten einlesen
    df = pd.read_csv(data_path, usecols=columns, dtype=dtype)
    
    # Daten filtern
    if only_big_cities:
        # print(f'Anzahl der Einträge vor dem Filtern: {df.shape[0]}')
        df = df[df['Population'] > 100000] # nur Städte mit mehr als 100.000 Einwohnern
        # print(f'Anzahl der Einträge nach dem Filtern: {df_big_cities.shape[0]}') # laut Paper 3.527 (passt)

    if n is None:
        df = df.reset_index()
        df.to_csv(samples_path, index=False)
        return True

    # Zufällige n Städte auswählen
    indices = df.index.values.tolist()
    rng = np.random.default_rng()
    sample_indices = rng.choice(indices, size=n, replace=False, shuffle=False) # wählt n zufällige Indizes aus, wobei jeder Index max. einmal vorkommen kann
    samples = df.loc[sample_indices]
    samples = samples.reset_index()
    # print(f'Anzahl der Einträge nach dem Auswählen: {samples.shape[0]}')

    # Speichern der Daten
    samples.to_csv(samples_path, index=False)
    return True


def load_samples(
        samples_path,
        dtype=dtype
        ):
    """
    Lädt einen Datensatz.
    
    Args:
        samples_path (str): Der Pfad der Datei.
        dtype: dict(zip(['Country', 'City', 'Population', 'Latitude', 'Longitude'], ['object', 'object', 'Int64', 'float64', 'float64'])): Typen der Spalten.
        
    Returns:
        Der Datensatz.
    """
    return pd.read_csv(samples_path, dtype=dtype)


def query(model, template_path, file_path, DEEPINFRA_API_KEY, samples, add_country=False):
    """
    Stellt Anfragen an ein LLM.

    Args:
        model (str): Die Modellbezeichnung von DeepInfra.
        template_path (str): Der Dateipfad von der Anfragenvorlage.
        file_path (str): Der Pfad der Datei, in der der Output gespeichert werden soll.
        DEEPINFRA_API_KEY (str): Der API-Key von DeepInfra.
        samples (df): Die Städte, für die die Anfrage durchgeführt werden soll.
        add_country: False (bool): Soll die Länder-Information hinzugefügt werden?
    """
    # Anfragen nur durchführen, wenn sie nicht schon gemacht wurden
    if not os.path.exists(file_path):
        print(f"Achtung! Die Datei {file_path} existiert bereits.\nDaher werden die zugehörigen Anfragen nicht ausgeführt. Es ist möglich, dass dieses Datei allerdings mit anderen Parametern und/oder Daten erzeugt wurde.")
        return
    
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    messages = [] # beinhaltet die Anfragen an das LLM
    answers = [] # beihnaltet die jeweiligen Antworten des LLMs

    # Fortschrittsanzeige erstellen und starten
    progress = progress_bar(samples.shape[0])
    progress.start()

    with open(template_path) as template_file:
        template = json.load(template_file)

        for _, row in samples.iterrows():
            location = None
            city = row['City']

            # Hinzufügen oder weglassen des Landes
            if add_country:
                country_code = row['Country']
                country = dict(pycountry.countries.lookup(country_code))['name']

                location = f"{city}, {country}"
            else:
                location = city

            # Zusammenbauen der Anfrage
            custom_message = copy.deepcopy(template)
            custom_message[1]['content'] = custom_message[1]['content'].format(location=location)

            messages.append(custom_message)

            # Anfrage
            chat_completion = openai.chat.completions.create(
                model=model,
                messages=custom_message,
            )

            answers.append(chat_completion.choices[0].message.content)

            progress.update()

    progress.stop()
    # Ergänzen der Daten mit den Anfragen
    result = samples.assign(Messages=pd.Series(messages).values, Answers=pd.Series(answers).values)

    # Exportieren und Speichern
    result.to_csv(file_path, index=False)
    

def query_save(model, template_path, file_path, DEEPINFRA_API_KEY, samples, add_country=False,
               continuous_save=True, continuous_save_n=10):
    """
    Stellt Anfragen an ein LLM.

    Args:
        model (str): Die Modellbezeichnung von DeepInfra.
        template_path (str): Der Dateipfad von der Anfragenvorlage.
        file_path (str): Der Pfad der Datei, in der der Output gespeichert werden soll.
        DEEPINFRA_API_KEY (str): Der API-Key von DeepInfra.
        samples (df): Die Städte, für die die Anfrage durchgeführt werden soll.
        add_country: False (bool): Soll die Länder-Information hinzugefügt werden?
        continuous_save: True (bool): Soll die Datei mehrmals zwischengespeichert werden?
        continuous_save_n: 10 (int): Anzahl der Schritte, zwischen dem speichern. 
    """
    # Vorbereitung, falls diese Datei noch nicht existiert (initialisieren der leeren Spalten)
    if not os.path.exists(file_path):
        df = samples
        df['Messages'] = None
        df['Answers'] = None
        df = df.astype({'Messages': 'object', 'Answers': 'object'})
        df.to_csv(file_path, index=False)
    
    # Datei laden und noch nicht gemachte Anfragen finden
    df = pd.read_csv(file_path, dtype=dtype)
    df = df.astype({'Messages': 'object', 'Answers': 'object'})
    indices = df[df[['Messages', 'Answers']].isnull().any(axis=1)].index.values.tolist()

    # Falls die Anfrage abgeschlossen wurde
    if len(indices) == 0:
        print(f"Die Datei {file_path} wurde vollständig erstellt!")
        return

    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # Fortschrittsanzeige erstellen und starten
    progress = progress_bar(len(indices))
    progress.start()

    with open(template_path) as template_file:
        template = json.load(template_file)

        for counter, index in enumerate(indices):
            
            # Zwischenspeichern
            if continuous_save and counter % continuous_save_n == 0 and counter > 0:
                df.to_csv(file_path, index=False)
            
            row = df.iloc[index]
            location = None
            city = row['City']

            # Hinzufügen oder weglassen des Landes
            if add_country:
                country_code = row['Country']
                country = dict(pycountry.countries.lookup(country_code))['name']

                location = f"{city}, {country}"
            else:
                location = city

            # Zusammenbauen der Anfrage
            custom_message = copy.deepcopy(template)
            custom_message[1]['content'] = custom_message[1]['content'].format(location=location)

            # Anfrage
            chat_completion = openai.chat.completions.create(
                model=model,
                messages=custom_message,
            )

            df.at[index, 'Messages'] = custom_message
            df.at[index, 'Answers'] = chat_completion.choices[0].message.content

            progress.update()

    progress.stop()

    # Exportieren und Speichern
    df.to_csv(file_path, index=False)


def process_row(args):
    """
    Process a single row: builds the request and queries the model.

    Returns a tuple (index, messages, answer)
    """
    (model, template, index, row, DEEPINFRA_API_KEY, add_country) = args
    location = None
    city = row['City']

    # Hinzufügen oder weglassen des Landes
    if add_country:
        country_code = row['Country']
        country = dict(pycountry.countries.lookup(country_code))['name']

        location = f"{city}, {country}"
    else:
        location = city

    # Deep copy the template
    custom_message = copy.deepcopy(template)
    custom_message[1]['content'] = custom_message[1]['content'].format(location=location)

    # Initialize openai object here (process-safe)
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # Make the request
    chat_completion = openai.chat.completions.create(
        model=model,
        messages=custom_message,
    )

    answer = chat_completion.choices[0].message.content
    return (index, custom_message, answer)


def query_save_parallel(model, template_path, file_path, DEEPINFRA_API_KEY, samples, add_country=False,
                        continuous_save=True, continuous_save_n=10, process_number=50):
    """
    Stellt Anfragen an ein LLM parallel mit multiprocessing Pool.
    """
    # Vorbereitung, falls diese Datei noch nicht existiert (initialisieren der leeren Spalten)
    if not os.path.exists(file_path):
        df = samples
        df['Messages'] = None
        df['Answers'] = None
        df = df.astype({'Messages': 'object', 'Answers': 'object'})
        df.to_csv(file_path, index=False)
    
    # Datei laden und noch nicht gemachte Anfragen finden
    df = pd.read_csv(file_path, dtype=dtype)
    df = df.astype({'Messages': 'object', 'Answers': 'object'})
    indices = df[df[['Messages', 'Answers']].isnull().any(axis=1)].index.values.tolist()

    # Falls die Anfrage abgeschlossen wurde
    if len(indices) == 0:
        print(f"Die Datei {file_path} wurde vollständig erstellt!")
        return

    # Load the template once
    with open(template_path) as template_file:
        template = json.load(template_file)

        # Prepare tasks for parallel execution
        tasks = []

        for index in indices:
            row = df.iloc[index]
            args = (model, template, index, row, DEEPINFRA_API_KEY, add_country)
            tasks.append(args)

        # Fortschrittsanzeige erstellen und starten
        progress = progress_bar(len(tasks), title=file_path)
        progress.start()

        # Use a pool of workers
        # Adjust processes according to your system
        # For large-scale I/O-bound tasks, you might consider a smaller or larger number of workers
        with Pool(processes=process_number) as pool:
            
            # We can use imap_unordered for results as they come in, enabling progress reporting
            for counter, (index, messages, answer) in enumerate(pool.imap_unordered(process_row, tasks)):

                # Zwischenspeichern
                if continuous_save and counter % continuous_save_n == 0 and counter > 0:
                    df.to_csv(file_path, index=False)

                df.at[index, 'Messages'] = messages
                df.at[index, 'Answers'] = answer

                progress.update()

    progress.stop()
    # Exportieren und Speichern
    df.to_csv(file_path, index=False)