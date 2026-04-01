from datetime import datetime
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import sigmf
from scipy import signal, constants


MIN_DB = -150
MAX_DB = -90

"""
Find all .sigmf-data files in the current working directory
and run the compression routine on each.
"""
def compress_all_sigmf_in_cwd(k,NFFT,NOVERLAP):

    #Check for .sigmf files in the current working directory
    cwd = os.getcwd()
    data_files = sorted(glob.glob(os.path.join(cwd, "*.sigmf-data")))

    if not data_files:
        print("[INFO] No .sigmf-data files found")
        return

    for data_path in data_files:
        base = data_path.replace(".sigmf-data", "")
        meta_path = base + ".sigmf-meta"

        if not os.path.exists(meta_path):
            print(f"[WARN] Missing meta file for {data_path}, skipping")
            continue

        print(f"[INFO] Compressing {os.path.basename(base)}")

        try:
            compressSigmf(
                filename=base,
                k=k,
                NFFT=NFFT,
                NOVERLAP=NOVERLAP,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {data_path}: {e}")

    print("[INFO] Compression complete")


    """
    Compress a .sigmf file pair into a relative velocity sparce representation in txt format.
    """
def compressSigmf(filename,k,NFFT,NOVERLAP):

    #Read the file's metadata 
    try: 
        datafile = sigmf.fromfile(filename)

        globalMetadata = datafile.get_global_info()
        captures = datafile.get_captures()
        annotations = datafile.get_annotations()

        sampleRate = globalMetadata.get('core:sample_rate','N/A')
        centerFreq = captures[0]['core:frequency']
        
        dtype_str = globalMetadata.get('core:datatype', 'cf32_le') # Default to complex float 32-bit little-endian
        if dtype_str == 'cf32_le':
            data_dtype = np.complex64
        elif dtype_str == 'ci16_le':
            data_dtype = np.int16

    except FileNotFoundError:
        print("Error: The file " + filename + ".sigmf-meta"+ " was not found.")

    #Read in the file's raw data
    rawData = np.fromfile(filename + ".sigmf-data", dtype=data_dtype)

    #Create STFT from the raw data
    f, t, STFT = signal.stft(rawData, fs=sampleRate, nperseg=NFFT, noverlap=NOVERLAP, return_onesided=False)

    STFTShift = np.fft.fftshift(STFT, axes=0)
    fShift = np.fft.fftshift(f)


    #Lock in the shift
    STFT = STFTShift
    f = fShift

    # Solve the optimizaiton per frame
    STFT = np.array(STFT)

    #Zero pad the data so that it is a perfect multiple of NFFT
    rawData = np.append(
    rawData,
    np.zeros((NFFT - len(rawData) % (NFFT - NOVERLAP)) % NFFT, dtype=complex)
    )
    
    #Determine how many frames of data there are
    numFrames = STFT.shape[1]

    outputVals = np.zeros((numFrames,k),dtype=complex)
    outputFreqs = np.zeros((numFrames,k))

    DCWidth = 5000
    killInds = np.where((np.abs(f) <= DCWidth))

    c = constants.c

    # Store the running median
    maxVals = np.zeros((numFrames,1))


    #Store the largest k values 
    for i in range(numFrames):
        workingRow = np.abs(STFT[:,i])

        #Ignore the DC Band
        workingRow[killInds] = 0

        #Identify the largest k values in the row and their indices
        largeInds = np.argpartition(workingRow, -k)[-k:]
        largeInds = largeInds[np.argsort(f[largeInds])]

        largeVals = STFT[largeInds,i]

        #Difference from the center frequency
        satFreq =  f[np.argmax(np.abs(workingRow))] 

        #Compute the relative velocity for the dominant frequencies
        relVels = c*f[largeInds]/centerFreq 



        maxVals[i] = satFreq 
        outputVals[i,:] = largeVals      
        #outputFreqs[i,:] = f[largeInds] #Store compressed frequencies
        outputFreqs[i,:] = relVels       #Store compressed relative velocities


    #Merge the frequency data with the complex values
    outputFile = np.column_stack((outputFreqs,outputVals))

    #Write compressed data to text file
    np.savetxt(filename+".txt", outputFile, fmt='%.10g', delimiter=' ')

    #Return values for reconstruction 
    return sampleRate, f, t


"""
Count number of complex samples in a text file written by np.savetxt,
where each line contains one complex sample (real imag).
"""
def get_num_samples(txt_path: str) -> int:

    count = 0
    with open(txt_path, "r") as f:
        for line in f:
            if line.strip():  # ignore empty lines
                count += 1
        #print('counting:')
        #print(count)
    return count


"""
For a given frequency band (list of records with same cf),
compute a stitched spectrogram over time.

Returns:
    f_final  : frequency axis (Hz)
    t_full   : time axis (seconds from t0_global)
    Sxx_full : 2D array [freq_bins x time_bins], in dB with DC notch applied
"""
def compute_band_spectrogram(records_band, t0_global, sample_rate, k, NFFT, NOVERLAP):

    # Sort by actual capture time
    records_band = sorted(records_band, key=lambda r: r["capture_time"])

    print('Record bands size ', len(records_band))

    f_axis = None
    Sxx_segments = []
    t_segments = []
    maxVel = 0

    for j, rec in enumerate(records_band):
        txt_path = rec["data"]
        cf = rec["cf"]
        t_capture = rec["capture_time"]

        # Read full file and decompress the data
        n_samps = get_num_samples(txt_path)
        if n_samps == 0:
            print(f"[WARN] No samples in {txt_path}, skipping")
            continue
        else:
            
            # Velocity axis (m/s)
            # Use observed max velocity to size axis conservatively
            v_lim = 5000
            v_max = v_lim

            #Number of velocity bins
            n_vel_bins = 500 
            v = np.linspace(-v_max, v_max, n_vel_bins)

            num_segments = np.floor(n_samps/k)

            data = np.loadtxt(txt_path,dtype=complex)
            numRows = data.shape[0]

            step = NFFT - NOVERLAP
            t = (np.arange(numRows)*step + NFFT/2) / sample_rate

            #Sort the values from the text file for reconstruction
            reconFreqs = data[:,0:k]
            reconVals = data[:,k:]
            sparseSTFT = np.zeros((len(v),numRows),dtype=complex)

            #Build the sparse STFT by placing the values at the appropriate frequency bins
            for i in range(numRows):
                indices = np.searchsorted(v, reconFreqs[i,:])
                indices = np.clip(indices,0, len(v)-1)
                
                sparseSTFT[indices,i] = reconVals[i,:]



        # dB scale 
        Sxx_db = 10.0 * np.log10(np.abs(sparseSTFT) + 1e-15)

        # Global time axis for this capture
        t_global = t

        # Keep frequency axis consistent
        if f_axis is None:
            f_axis = v
        else:
            if len(v) != len(f_axis) or not np.allclose(v, f_axis, rtol=0, atol=1.0):
                print(
                    f"[WARN] Frequency axis mismatch in {os.path.basename(txt_path)}, "
                    "skipping this capture for stitched plot."
                )
                continue

        
        Sxx_segments.append(Sxx_db)
        t_segments.append(t_global)


        ## SAVE NUMPY OUTPUTS
        out_name = f"r{j}_waterfall_CF_{cf/1e9:.3f}GHz.png"
        out_tag = f"r{j}_waterfall_CF_{cf/1e9:.3f}GHz.npy"
        

        ######## 
        np.save(out_tag,sparseSTFT.T)
        np.save("time_"+ out_tag,t_segments)
        np.save('rel_vel_'+out_tag,f_axis)


        #Convert relative time to absolute timestamps
        stop_np = np.datetime64(t_capture)                          
        offset_ns = (np.asarray(t) * 1e9).astype(np.int64)
        start_time = stop_np - offset_ns[-1].astype('timedelta64[ns]')
        datetimestamps = start_time + offset_ns.astype('timedelta64[ns]')
        np.save('datetime_updated_' + out_tag, datetimestamps)

        
        # Save per-band waterfall plot 
        extent = [f_axis[0], f_axis[-1], t[0], t[-1]]
 
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            Sxx_db.T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="turbo",
            interpolation="bilinear",
            vmin=MIN_DB,
            vmax=MAX_DB,
        )
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Waterfall: CF ≈ {cf/1e9:.3f} GHz")
        fig.tight_layout()
        fig.savefig(out_name, dpi=200)
        plt.close(fig)
        print(f"  Saved r{j} plot: {out_name}")

    if not Sxx_segments:
        return None, None, None
    

    #Define the relative velocity axis based on the maximum velocity seen 
    #f_axis = np.linspace(-maxVel-1000,maxVel+1000,len(f_axis))
    f_axis = v

    # Stitch along time axis
    Sxx_full = np.concatenate(Sxx_segments, axis=1)  # [freq_bins x total_time_bins]
    t_full = np.concatenate(t_segments)

    # # Notch out DC (center of freq axis)
    # mid_index = Sxx_full.shape[0] // 2
    # floor_value = np.percentile(Sxx_full, 5)
    # lo = max(0, mid_index - NOTCH_WIDTH)
    # hi = min(Sxx_full.shape[0], mid_index + NOTCH_WIDTH)
    # Sxx_full[lo:hi, :] = floor_value

    return f_axis, t_full, Sxx_full



"""
Scan current working directory for .txt files and return a list of dicts:
{
    'data': txt_path,
    'cf': center_freq (GHz),
    'capture_time': datetime
}
"""
def find_txt_records():

    txt_files = glob.glob("*.txt")
    records = []

    for txt_path in sorted(txt_files):
        filename = os.path.basename(txt_path)

        cf, capture_time = parse_filename_metadata(filename)
        if cf is None or capture_time is None:
            print(f"[WARN] Could not parse metadata from {filename}, skipping")
            continue

        records.append(
            {
                "data": txt_path,
                "cf": cf,
                "capture_time": capture_time,
            }
        )

    return records

"""
Extract center frequency (GHz) and capture time from filename:
r001_f11.200GHz_20251217T204137.txt
"""
def parse_filename_metadata(filename: str):
    import re

    freq_match = re.search(r'_f([\d.]+)GHz_', filename)
    time_match = re.search(r'_(\d{8}T\d{6})(?:\D|$)', filename)

    if not freq_match or not time_match:
        return None, None

    cf = float(freq_match.group(1))*1e9
    capture_time = datetime.strptime(time_match.group(1), "%Y%m%dT%H%M%S")

    return cf, capture_time


# Defining main function
def main():

    ### USER DEFINED INPUTS ###
    sample_rate = 500000.0

    #Specify the number of frequencies to keep
    k = 10 

    #Specify the STFT Parameters
    NFFT = 1024
    NOVERLAP = 512

    #Compress all .sigmf files in the current working directory (comment out if already done)
    compress_all_sigmf_in_cwd(k,NFFT,NOVERLAP)


    records = find_txt_records()
    if not records:
        print(f"No valid compressed .txt captures found in working directory")
        return

    print(f"Found {len(records)} compressed .txt captures")

    # Global earliest time for time-zero
    t0_global = min(r["capture_time"] for r in records)
    print(f"Global start time: {t0_global}")

    # Group by center frequency
    bands = {}
    for r in records:
        bands.setdefault(r["cf"], []).append(r)

    cfs = sorted(bands.keys())
    print(f"Discovered {len(cfs)} distinct center frequencies")
    print(cfs)

    band_results = {}  # cf -> (f_axis, t_full, Sxx_full)

 
    # ---- Compute per-band stitched spectrograms ----
    for cf in cfs:
        print(f"\n=== Processing band CF = {cf/1e9:.3f} GHz ===")
        f_axis, t_full, Sxx_full = compute_band_spectrogram(bands[cf], t0_global, sample_rate, k, NFFT, NOVERLAP)
        if f_axis is None:
            print(f"  [WARN] No data for CF {cf/1e9:.3f} GHz")
            continue

        band_results[cf] = (f_axis, t_full, Sxx_full)





if __name__=="__main__":
    main() 