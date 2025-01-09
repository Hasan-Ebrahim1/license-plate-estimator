import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function Title({ text }) {
  return <h1 className="title">{text}</h1>;
}

function LicensePlateInput({ value, onChange, onGo, disableGo }) {
  return (
    <div className="license-input-container">
      <label>Enter your license plate number:</label>
      <div className="input-button-row">
        <input
          type="text"
          value={value}
          onChange={onChange}
          placeholder="Enter 3-6 digits"
        />
        <button onClick={onGo} disabled={disableGo}>Go</button>
      </div>
    </div>
  );
}

function PriceDisplay({ loading, price }) {
  return (
    <div className="price-display">
      {loading ? <span>Loading...</span> : <span>{Math.round(price)} BHD</span>}
    </div>
  );
}

function App() {
  const [licensePlateNumber, setLicensePlateNumber] = useState('');
  const [estimatedPrice, setEstimatedPrice] = useState(0);
  const [loading, setLoading] = useState(false);

  const onLicensePlateChange = (event) => {
    const cleanedValue = event.target.value.replace(/\D/g, '').slice(0, 6);
    setLicensePlateNumber(cleanedValue);
  };

  const goButton = () => {
    if (licensePlateNumber.length < 3) {
      alert('Please enter at least 3 digits.');
      return;
    }

    setLoading(true);
    axios
      .post("https://license-plate-estimator.onrender.com/predict", { plate_number: licensePlateNumber })
      .then((res) => {
        setEstimatedPrice(res.data.predicted_price);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error caught: ", err);
        setLoading(false);
      });
  };

  return (
    <div className="App">
      <Title text="Bahrain License Plate Price Estimator" />
      <div className="middle-part">
        <LicensePlateInput 
          value={licensePlateNumber}
          onChange={onLicensePlateChange}
          onGo={goButton}
          disableGo={licensePlateNumber.length < 3}
        />
        <PriceDisplay loading={loading} price={estimatedPrice} />
      </div>
    </div>
  );
}

export default App;
