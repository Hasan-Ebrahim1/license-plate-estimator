import './App.css'
import React from 'react'
import { useState, useEffect } from 'react';
import axios from 'axios'

function App() {

  const [licensePlateNumber, setLicensePlateNumber] = useState('');
  const [estimatedPrice, setEstimatedPrice] = useState(0);
  const [loading, setLoading] = useState(false);

  const onLicensePlateChange = (event) => {
    setLicensePlateNumber(event.target.value)
    console.log(licensePlateNumber)
  }

  const goButton = () => {
    setLoading(true)
    axios.post("https://license-plate-estimator.onrender.com/predict", { "plate_number" : licensePlateNumber})
    .then((res)=> {
      const response = res.data;
      console.log(response[0]);
      setEstimatedPrice(response.predicted_price)
      setLoading(false)
    })
    .catch((err) =>{
      console.log("error caught: ", err)
    })
    
  }

  return (
    <div className="App">
      <div className="title">
        Bahrain License Plate Price Estimator
      </div>
      <div className='middle-part'>
        <div>
          Enter your license plate number:
        </div>
        <div>
          <div>
            <input
              value={licensePlateNumber}
              onChange={onLicensePlateChange}
            />
            <button
              onClick={goButton}
            >
              Go
            </button>
          </div>
          {loading ?
          
            <span> Loading... </span>
            :
            <span> {Math.round(estimatedPrice)} BHD</span>

          }
        </div>
      </div>
    </div>
  );
}

export default App;
