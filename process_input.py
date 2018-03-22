import os, sys, warnings, multiprocessing
from tqdm import tqdm
import scipy.io.wavfile
import soundfile as sf
import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
from rpy2.robjects.packages import importr
warbleR = importr("warbleR")
seewave = importr("seewave")
tuneR = importr("tuneR")

def process_audio(audio, rate, names, audio_type='flac'):
    try:
        if audio_type == 'flac':
            audio = audio.reshape(-1, 1)

        #calculate spectr of a sound file
        spec_result = np.array(seewave.meanspec(audio, rate, plot='FALSE'))
        freq = spec_result[:, 0]
        amp = spec_result[:, 1]

        # Amplitude
        amp = amp / np.sum(amp)
        cumamp = np.cumsum(amp)

        # RESULTS

        mean = np.sum(amp*freq)
        std = np.sqrt(np.sum(amp*((freq-mean)**2)))
        sem = std/np.sqrt(amp.size)
        mode = freq[np.argmax(amp)]

        median = freq[np.size(cumamp[cumamp <= 0.5])+1]
        Q25 = freq[np.size(cumamp[cumamp <= 0.25])+1]
        Q75 = freq[np.size(cumamp[cumamp <= 0.75])+1]
        IQR = Q75 - Q25
        cent = np.sum(freq * amp)
        z = np.sum(amp - np.mean(amp))
        w = np.std(amp)
        skew = (np.sum((amp - np.mean(amp))**3)/(amp.size - 1))/w**3
        kurt = (np.sum((amp - np.mean(amp))**4)/(amp.size - 1))/w**4

        #spectral entropy
        sh = seewave.sh(spec_result)
        ent = sh[0]

        #spectral flatness
        sfm = seewave.sfm(spec_result)
        sfm = sfm[0]

        #fpeaks
        fpeak = np.array(seewave.fpeaks(spec_result, f=rate, nmax=3, plot='FALSE'))[0,0]

        #fundamental frequancy
        freq_fun = np.array(seewave.fund(audio, f = rate, ovlp = 50, threshold = 15, plot = 'FALSE'))
        freq_fun = freq_fun[~np.isnan(freq_fun).any(axis=1)][:, 1]

        fund_freq = np.mean(freq_fun)
        fund_min = np.min(freq_fun)
        fund_max = np.max(freq_fun)

        #dominant frequancy 

        freq_dom = np.array(seewave.dfreq(audio, f = rate, ovlp = 50, threshold = 15, plot = 'FALSE'))
        freq_dom = freq_dom[~np.isnan(freq_dom).any(axis=1)][:, 1]
        dom_freq = np.mean(freq_dom)
        dom_min = np.min(freq_dom)
        dom_max = np.max(freq_dom)

        #dfrange
        dfrange = dom_max - dom_min

        #modindx
        modindx = np.sum(np.abs(freq_dom[1:] - freq_dom[:-1])) / dfrange if dfrange else np.nan

        audio_stats = [mean, std, median, Q25, Q75, IQR, skew, kurt, ent, sfm, mode, cent, fpeak, fund_freq, fund_min, fund_max, dom_freq, dom_min, dom_max, dfrange, modindx]

        return dict(zip(names, audio_stats))
    except (Exception, RuntimeWarning) as ex:

        print(str(ex)) 
        return dict(zip(names, len(names) * [None])) 

def process_files(features, ids, procnum, directory, filenames, names, files_type='flac'):
        fet = []
        idx = []
        for filename in tqdm(filenames):
            if files_type == 'flac':
                with open('{}/{}'.format(directory, filename), 'rb') as f:
                    data, samplerate = sf.read(f)
                f.close()
                idx.append(int(filename.partition('-')[0]))
            elif files_type == 'wav':
                data = tuneR.readWave('{}/{}'.format(directory, filename),  units = "seconds")
                samplerate = data.slots['samp.rate'][0]
                idx.append(filename.split('.')[0])
            else:
                print('Method support only FLAC or WAV files')
                raise TypeError
            fet.append(process_audio(data, samplerate, names, audio_type=files_type))

        features[procnum] = fet
        ids[procnum] = idx

def form_data_frame(df, columns, directory, filenames, n_jobs=1, files_type='flac'):
    manager = multiprocessing.Manager()
    dict_features = manager.dict()
    dict_ids = manager.dict()

    jobs = []
    files_per_proc = len(filenames) // n_jobs + 1
    for proc_index in range(n_jobs):
        process = multiprocessing.Process(target=process_files,
                                          args=(dict_features, dict_ids, proc_index, directory, 
                                                filenames[proc_index * files_per_proc: (proc_index+1) * files_per_proc], columns, files_type))
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


    id_persons = []
    features = []
    for i in range(n_jobs):
        id_persons += dict_ids[i]
        features += dict_features[i]
    
    return features, id_persons

def make_csv_from_audios(speakers_file, audios_directory, to_csv_file):
    data = pd.read_table(speakers_file, skiprows=11)
    array = []
    for row in data.iterrows():
        array.append(row[1].values[0].split('|'))
    data = pd.DataFrame(array)
    df = data.drop(data.columns[2::], axis=1)
    df.columns = ['id', 'sex']
    df['sex'] = pd.factorize(df['sex'])[0]
    df['id'] = df['id'].astype(np.int16)
    df = df.set_index('id')
    filenames = []
    pattern = audios_directory

    taken_persons = set()
    directory = os.fsencode(pattern)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.splitext(filename)[1][1:] == 'flac':
            id_person = int(filename.partition('-')[0])
            if id_person not in taken_persons:
                taken_persons.add(id_person)
                filenames.append(filename)
    names = ['meanfr', 'std', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'ent', 'sfm', 'mode', 'cent', 'fpeak',
         'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
    features, id_persons = form_data_frame(df, names, pattern, filenames, n_jobs=multiprocessing.cpu_count())
    
    persons = pd.DataFrame.from_items([('id', id_persons)])
    persons = persons.set_index('id')
    persons = persons.join(df)
    full_data = pd.concat([persons, pd.DataFrame(features, index=persons.index)], axis=1)
    full_data = full_data.fillna(0)
    
    full_data.to_csv(to_csv_file)
    
def make_csv_from_wav(speakers_file, audios_directory, to_csv_file):
    df = pd.read_table(speakers_file, sep=';')
    df['sex'] = 1 - pd.factorize(df['sex'])[0]
    
    filenames = []
    pattern = audios_directory

    taken_persons = set()
    directory = os.fsencode(pattern)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.splitext(filename)[1][1:] == 'wav':
            name_person = os.path.splitext(filename)[0]
            if name_person not in taken_persons:
                taken_persons.add(name_person)
                filenames.append('{}'.format(filename))
    names = ['meanfr', 'std', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'ent', 'sfm', 'mode', 'cent', 'fpeak',
         'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
    features, fileperson = form_data_frame(df, names, pattern, filenames, 
                                       n_jobs=multiprocessing.cpu_count(), files_type='wav')
    persons = pd.DataFrame.from_items([('filename', fileperson)])
    persons = pd.merge(persons, df, how='inner')
    full_data = pd.concat([persons, pd.DataFrame(features)], axis=1).drop('filename', axis=1)
    full_data = full_data.fillna(0)
    
    full_data.to_csv(to_csv_file)
    
def make_csv_from_specan(speakers_file, audios_directory, to_csv_file):
    df = pd.read_csv(speakers_file, sep=';')
    df['sex'] = 1 - pd.factorize(df['sex'])[0]
    filenames = []
    pattern = audios_directory

    taken_persons = set()
    directory = os.fsencode(pattern)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.splitext(filename)[1][1:] == 'wav':
            name_person = os.path.splitext(filename)[0]
            if name_person not in taken_persons:
                taken_persons.add(name_person)
                filenames.append('{}'.format(filename))
    
    df_specan = pd.DataFrame({'sound.files': filenames, 'selec': np.ones(np.size(filenames)), 
                              'start': np.zeros(np.size(filenames)), 'end': 30*np.ones(np.size(filenames))})
    specan_data = warbleR.specan(df_specan)
    specan_df = pd.DataFrame(np.array(specan_data).T, columns=specan_data.colnames)
    specan_df['filename'] = [filename[:-4] for filename in filenames]
    specan_df = pd.merge(specan_df, df, on='filename')
    specan_df = specan_df.drop(['sound.files', 'selec', 'duration', 
                                'time.median', 'time.Q25', 'time.Q75', 'time.IQR',
                                 'startdom', 'enddom','dfslope', 'filename'], axis=1)
    
    specan_df.to_csv(to_csv_file)