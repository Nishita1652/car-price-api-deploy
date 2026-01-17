import React, { useState, useEffect } from 'react';
import './App.css';
import { CAR_MODEL_MAPPING } from './constants';
import tireImage from './assets/tire.png';

const CURRENT_YEAR = new Date().getFullYear();
const API_URL = 'https://car-price-api-deploy.onrender.com/selling_price';
const YEARS_LIST = Array.from({ length: 26 }, (_, i) => CURRENT_YEAR - i);

function App() {
  const [formData, setFormData] = useState({
    brand: '', model: '', year: CURRENT_YEAR, seats: 5,
    km_driven_input: '', mileage: '', engine: '',
    max_power: '', fuel_type: 'Petrol', transmission_type: 'Manual', seller_type: 'Individual'
  });
  const [prediction, setPrediction] = useState(null);
  const [displayPrice, setDisplayPrice] = useState(0);
  const [loading, setLoading] = useState(false);
  const [isBrandModalOpen, setIsBrandModalOpen] = useState(false);
  const [isResultModalOpen, setIsResultModalOpen] = useState(false);
  const [kmAdjustment, setKmAdjustment] = useState(0);
  const [marketInsight, setMarketInsight] = useState("");

  const getLogoPath = (brand) => {
    if (!brand) return "";
    const fileName = brand.toLowerCase().replace(/\s+/g, '-');
    return `/src/assets/logos/${fileName}.svg`;
  };

  // Rolling Counter Logic (Previous to New Price)
useEffect(() => {
    if (prediction > 0) {
      let start = displayPrice; 
      const end = prediction;
      const duration = 500; 
      const startTime = performance.now();

      const animate = (currentTime) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const currentVal = Math.floor(start + (end - start) * progress);
        setDisplayPrice(currentVal);

        if (progress < 1) {
          requestAnimationFrame(animate);
        } else {
          setDisplayPrice(end);
        }
      };
      requestAnimationFrame(animate);
    }
  }, [prediction]);

  // Fixed Slider useEffect
  useEffect(() => {
    if (isResultModalOpen && formData.km_driven_input) {
      const delayDebounceFn = setTimeout(() => {
        const adjustedKm = parseFloat(formData.km_driven_input) + kmAdjustment;
        updatePrice(adjustedKm);
      }, 350); 
      return () => clearTimeout(delayDebounceFn);
    }
  }, [kmAdjustment]);

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setFormData(prev => ({ ...prev, [id]: value }));
  };

  const updateSeats = (delta) => {
    setFormData(prev => {
      const newSeats = prev.seats + delta;
      return (newSeats >= 2 && newSeats <= 8) ? { ...prev, seats: newSeats } : prev;
    });
  };

const updatePrice = async (adjustedKm) => {
  // Guard clause: Don't call API if essential data is missing
  if (!formData.brand || !formData.model) return;

  try {
    const payload = {
      brand: formData.brand,
      model: formData.model,
      year: parseInt(formData.year),
      km_driven: adjustedKm,
      fuel_type: formData.fuel_type,
      seller_type: formData.seller_type,
      transmission_type: formData.transmission_type,
      mileage_cleaned: parseFloat(formData.mileage) || 18,
      engine_cleaned: parseFloat(formData.engine) || 1200,
      max_power_cleaned: parseFloat(formData.max_power) || 85,
      seats: parseInt(formData.seats),
      vehicle_age: CURRENT_YEAR - parseInt(formData.year)
    };

    console.log("Sending Slider Update Payload:", payload);

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    console.log("Slider Update Success:", data);

    if (data.predicted_price_inr) {
      setPrediction(Math.round(data.predicted_price_inr));
      setMarketInsight(generateInsight(adjustedKm, formData.year, formData.brand));
    }
  } catch (err) {
    console.error("Slider API Error:", err);
  }
};

  const getPrediction = async () => {
    if (!formData.brand || !formData.km_driven_input) return alert("Select Brand and KM.");
    setLoading(true);
    setPrediction(null);
    setDisplayPrice(0);
    setKmAdjustment(0);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          km_driven: parseFloat(formData.km_driven_input),
          vehicle_age: CURRENT_YEAR - parseInt(formData.year),
          mileage_cleaned: parseFloat(formData.mileage) || 18,
          engine_cleaned: parseFloat(formData.engine) || 1200,
          max_power_cleaned: parseFloat(formData.max_power) || 85,
        })
      });
      const data = await response.json();
      setTimeout(() => {
        setPrediction(Math.round(data.predicted_price_inr));
        setLoading(false);
        setIsResultModalOpen(true);
        setMarketInsight(generateInsight(parseFloat(formData.km_driven_input), formData.year, formData.brand));
      }, 1500);
    } catch (err) { setLoading(false); }
  };
  
  const generateInsight = (km, year, brand) => {
  const age = CURRENT_YEAR - year;
  if (km > 100000) return "High mileage is the primary factor lowering this valuation.";
  if (age > 12) return "Vehicle age may affect financing options for buyers.";
  if (['Audi', 'BMW', 'Mercedes-Benz', 'Jaguar', 'Porsche'].includes(brand)) 
    return "Luxury brand status helps maintain a strong base resale value.";
  return "This model currently shows stable demand in the pre-owned market.";
};

  return (
    <div className="main-container">
      <div className="content-wrapper">
        <div className="card">
          <h1 className="form-title">Car Price Estimator</h1>
          
          <div className="input-grid">
            <div className="form-group full-width">
              <label>Vehicle Brand</label>
              <div className="brand-selector-box" onClick={() => setIsBrandModalOpen(true)}>
                <div className="selected-content">
                   {formData.brand && <img src={getLogoPath(formData.brand)} className="tiny-logo" alt="" />}
                   <span>{formData.brand || "Click to Select Brand"}</span>
                </div>
                <span className="chevron">▼</span>
              </div>
            </div>

            <div className="form-group">
              <label>Model</label>
              <select id="model" value={formData.model} onChange={handleInputChange}>
                <option value="">Select Model</option>
                {(CAR_MODEL_MAPPING[formData.brand] || []).map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>

            <div className="form-group">
              <label>Year</label>
              <select id="year" value={formData.year} onChange={handleInputChange}>
                {YEARS_LIST.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>

            <div className="form-group"><label>Mileage (kmpl)</label><input type="number" id="mileage" placeholder="eg: 40" onChange={handleInputChange} /></div>
            <div className="form-group"><label>Engine (CC)</label><input type="number" id="engine" placeholder="eg: 300" onChange={handleInputChange} /></div>
            <div className="form-group"><label>Max Power (bhp)</label><input type="number" id="max_power" placeholder="eg: 500" onChange={handleInputChange} /></div>

            <div className="form-group">
              <label>Seats</label>
              <div className="counter-row">
                <button className="cnt-btn" onClick={() => updateSeats(-1)}>-</button>
                <span className="cnt-val">{formData.seats}</span>
                <button className="cnt-btn" onClick={() => updateSeats(1)}>+</button>
              </div>
            </div>

            <div className="form-group full-width"><label>KM Driven</label><input type="number" id="km_driven_input" placeholder="eg: 5000" value={formData.km_driven_input} onChange={handleInputChange} /></div>

            {/* RESTORED SELLER TYPE */}
            <div className="form-group"><label>Seller Type</label>
              <select id="seller_type" value={formData.seller_type} onChange={handleInputChange}>
                <option value="Individual">Individual</option>
                <option value="Dealer">Dealer</option>
              </select>
            </div>

            <div className="form-group"><label>Fuel</label>
              <select id="fuel_type" value={formData.fuel_type} onChange={handleInputChange}>
                <option value="Petrol">Petrol</option>
                <option value="Diesel">Diesel</option>
                <option value="CNG">CNG</option>
                <option value="LPG">LPG</option>
              </select>
            </div>
            <div className="form-group full-width"><label>Transmission</label>
              <select id="transmission_type" value={formData.transmission_type} onChange={handleInputChange}>
                <option value="Manual">Manual</option><option value="Automatic">Automatic</option>
              </select>
            </div>
          </div>

          <button className="predict-btn" onClick={getPrediction} disabled={loading}>
            {loading ? 'Analyzing...' : 'Calculate Resale Price'}
          </button>
          <p className='disclaimer'>First use might take some time</p>
        </div>
      </div>

      {/* SOLID POPUP RESULT */}
      {isResultModalOpen && (
        <div className="modal-overlay">
          <div className="popup-card">
            <button className="close-x" onClick={() => setIsResultModalOpen(false)}>&times;</button>
            <div className="res-header">
                <img src={getLogoPath(formData.brand)} className="res-logo" alt="" />
                <p className="res-subtitle">{formData.brand} {formData.model}</p>
            </div>
            <h2 className="res-price">₹ {displayPrice.toLocaleString('en-IN')}</h2>
            
            <div className="slider-box-container">
              <div className="slider-meta">
                <span>-5000km</span>
                <strong>{kmAdjustment > 0 ? `+${kmAdjustment}` : kmAdjustment} km</strong>
                <span>+5000km</span>
              </div>
              <input type="range" min="-5000" max="5000" step="500" value={kmAdjustment} onChange={(e) => setKmAdjustment(parseInt(e.target.value))} className="km-custom-slider" />
            </div>
            <div className="modal-insight">
                <p>💡 <strong>Market Insight:</strong> {marketInsight}</p>
            </div>
          </div>
        </div>
      )}

      {/* LOADER */}
      {loading && (
        <div className="modal-overlay">
          <div className="loading-content">
            <img src={tireImage} className="spinning-tire" alt=""/>
            <p className="load-text">Fetching Market Data...</p>
          </div>
        </div>
      )}

      {/* BRAND MODAL */}
      {isBrandModalOpen && (
        <div className="modal-overlay" onClick={() => setIsBrandModalOpen(false)}>
          <div className="brand-picker" onClick={e => e.stopPropagation()}>
            <div className="brand-grid-layout">
              {Object.keys(CAR_MODEL_MAPPING).sort().map(b => (
                <div key={b} className="brand-option" onClick={() => {setFormData(p=>({...p, brand:b, model:''})); setIsBrandModalOpen(false)}}>
                  <img src={getLogoPath(b)} className="brand-modal-logo" alt="" />
                  <span>{b}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;