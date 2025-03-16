<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Spam Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 700px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2 {
            color: #333;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            padding: 5px 0;
        }
        .emoji {
            font-size: 1.2em;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>📧 Email Spam Detector using Logistic Regression</h1>

        <h2>🔹 About the Project</h2>
        <p>This is a simple Spam Detection project using <strong>Logistic Regression</strong>. It analyzes email messages and predicts whether they are <strong>Spam (1) or Ham (0)</strong>.</p>

        <h2>🔹 How It Works?</h2>
        <ul>
            <li>1️⃣ Loads the dataset (spam & ham emails).</li>
            <li>2️⃣ Preprocesses the text (removes stopwords, converts to vectors).</li>
            <li>3️⃣ Trains a Logistic Regression model on the data.</li>
            <li>4️⃣ Predicts if a new email is Spam or Not.</li>
        </ul>

        <h2>🔹 Technologies Used</h2>
        <ul>
            <li>✅ Python 🐍</li>
            <li>✅ Scikit-learn 🤖</li>
            <li>✅ Pandas & NumPy 📊</li>
            <li>✅ Matplotlib 📉</li>
        </ul>
    </div>

</body>
</html>
