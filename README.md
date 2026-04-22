<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECG Arrhythmia Classification</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }
        h1, h2, h3 { color: #2c3e50; }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            border: 1px solid #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th { background-color: #f8f8f8; }
        hr { border: 0; border-top: 1px solid #eee; margin: 30px 0; }
    </style>
</head>
<body>

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
