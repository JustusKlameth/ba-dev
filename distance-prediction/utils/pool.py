import pycountry
import copy
import json
# Assume openai>=1.0.0
from openai import OpenAI
from utils.progress_bar import progress_bar
from multiprocessing import Pool
import os
import pandas as pd


columns = ['Country', 'City', 'Population', 'Latitude', 'Longitude'] # Spalten, die beachtet werden sollen
dtype = dict(zip(columns, ['object', 'object', 'Int64', 'float64', 'float64'])) # Datentypen der Spalten


def process_row(args):
    (model, template, index, row, DEEPINFRA_API_KEY, add_country) = args
    
    location_1 = None
    city_1 = row['City_1']

    location_2 = None
    city_2 = row['City_2']

    # Hinzufügen oder weglassen des Landes
    if add_country:
        country_code_1 = row['Country_1']
        country_1 = dict(pycountry.countries.lookup(country_code_1))['name']
        location_1 = f"{city_1}, {country_1}"

        country_code_2 = row['Country_2']
        country_2 = dict(pycountry.countries.lookup(country_code_2))['name']
        location_2 = f"{city_2}, {country_2}"
    else:
        location_1 = city_1
        location_2 = city_2

    # Zusammenbauen der Anfrage
    custom_message = copy.deepcopy(template)
    custom_message[1]['content'] = custom_message[1]['content'].format(location_1=location_1, location_2=location_2)

    # Initialize openai object
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
        return df

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
    return df