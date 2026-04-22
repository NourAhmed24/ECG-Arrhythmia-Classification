 <h1>ECG Arrhythmia Classification using 1D CNN</h1>
    <p>A TensorFlow implementation of a 1D Convolutional Neural Network for classifying cardiac arrhythmias from ECG signals using the MIT-BIH Arrhythmia Database.</p>

   <h2>Overview</h2>
    <p>This project implements a 1D CNN that classifies ECG beats into 5 categories (N, S, V, F, Q) following the AAMI standard, achieving state-of-the-art performance on the MIT-BIH dataset.</p>

   <table>
        <tr>
            <th>Class</th>
            <th>Description</th>
            <th>Clinical Significance</th>
        </tr>
        <tr>
            <td>N</td>
            <td>Normal beat</td>
            <td>Healthy</td>
        </tr>
        <tr>
            <td>S</td>
            <td>Supraventricular ectopic</td>
            <td>Moderate concern</td>
        </tr>
        <tr>
            <td>V</td>
            <td>Ventricular ectopic</td>
            <td>High risk</td>
        </tr>
        <tr>
            <td>F</td>
            <td>Fusion beat</td>
            <td>High risk</td>
        </tr>
        <tr>
            <td>Q</td>
            <td>Unknown/Pacemaker</td>
            <td>Special case</td>
        </tr>
    </table>

   <hr>

 
 

  <h3>2. Download and preprocess data</h3>
    <pre>from src.data_preprocessing import prepare_data
X_train, X_test, y_train, y_test, weights, le = prepare_data(download=True)</pre>


   <h2>Requirements</h2>
    <ul>
        <li>Python 3.8+</li>
        <li>TensorFlow 2.10+</li>
        <li>4GB RAM minimum</li>
        <li>No GPU required (CPU training ~30-60 min)</li>
    </ul>

  <h2>Dataset</h2>
    <p>MIT-BIH Arrhythmia Database: <a href="https://physionet.org/content/mitdb/1.0.0/">https://physionet.org/content/mitdb/1.0.0/</a></p>

   <h2>References</h2>
    <ol>
        <li>Hannun et al. (2019). Cardiologist-level arrhythmia detection with a deep neural network.</li>
        <li>Kachuee et al. (2018). ECG Heartbeat Classification: A Deep Transferable Representation.</li>
    </ol>

</body>
</html>
```
