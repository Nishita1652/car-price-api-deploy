import React, { useState } from 'react';
import './App.css';
import { CAR_MODEL_MAPPING } from './constants';

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
  const [loading, setLoading] = useState(false);
  const [sliderValue, setSliderValue] = useState(0);
  const [lastSuccessfulFeatures, setLastSuccessfulFeatures] = useState(null);
  const [isBrandModalOpen, setIsBrandModalOpen] = useState(false);
  const [marketInsight, setMarketInsight] = useState("");
  const isFormInvalid = 
  !formData.brand || 
  !formData.model || 
  !formData.km_driven_input || 
  !formData.engine || 
  !formData.mileage || 
  !formData.max_power;

  const formatNumber = (val) => val ? parseInt(val).toLocaleString('en-IN') : "";

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setFormData(prev => ({ ...prev, [id]: value }));
  };
  const getLogoPath = (brand) => {
  if (!brand) return "";
  const fileName = brand.toLowerCase().replace(/\s+/g, '-');
  return `/src/assets/logos/${fileName}.svg`;
};

  const selectBrand = (brandName) => {
    setFormData(prev => ({ ...prev, brand: brandName, model: '' }));
    setIsBrandModalOpen(false);
  };

  const updateSeats = (delta) => {
    setFormData(prev => {
      const newSeats = prev.seats + delta;
      return (newSeats >= 2 && newSeats <= 6) ? { ...prev, seats: newSeats } : prev;
    });
  };

  const generateInsight = (km, year, brand) => {
    if (km > 100000) return "High mileage is the primary factor lowering this valuation.";
    if (CURRENT_YEAR - year > 12) return "Vehicle age may affect financing options for potential buyers.";
    if (['Audi', 'BMW', 'Mercedes-Benz'].includes(brand)) return "Luxury brand status maintains a strong base resale value.";
    return "Market demand is currently stable for this category of vehicle.";
  };

  const getPrediction = async () => {
    if (!formData.brand || !formData.km_driven_input) {
      alert("Please select a Brand and enter KM Driven.");
      return;
    }
    setLoading(true);
    setPrediction(null);
    try {
      const payload = {
        km_driven: parseFloat(formData.km_driven_input),
        seller_type: formData.seller_type,
        fuel_type: formData.fuel_type,
        transmission_type: formData.transmission_type,
        mileage_cleaned: parseFloat(formData.mileage) || 18,
        engine_cleaned: parseFloat(formData.engine) || 1200,
        max_power_cleaned: parseFloat(formData.max_power) || 85,
        seats: parseInt(formData.seats),
        brand: formData.brand,
        model: formData.model,
        vehicle_age: CURRENT_YEAR - parseInt(formData.year)
      };

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      const price = Math.round(data.predicted_price_inr);
      setPrediction(price);
      setLastSuccessfulFeatures({ ...payload, predicted_price_inr: price });
      setSliderValue(payload.km_driven);
      setMarketInsight(generateInsight(payload.km_driven, formData.year, formData.brand));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-container">
      <div className="content-wrapper">
        <div className="card form-column">
          <h1 className="form-title">Car Price Estimator</h1>
          <div className="input-grid">
            <div className="grid-half">
              <div className="form-group">
                <label>Brand</label>
                <div className="brand-selector-trigger" onClick={() => setIsBrandModalOpen(true)}>
                  <div className="selected-brand-content">
                    {formData.brand && <img src={getLogoPath(formData.brand)} className="tiny-logo" />}
                    <strong>{formData.brand || "Select Brand"}</strong>
                  </div>
                  <svg className="chevron-icon" width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M5 7.5L10 12.5L15 7.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </div>
              </div>
              <div className="form-group">
                <label>Model</label>
                <select id="model" value={formData.model} onChange={handleInputChange}>
                  <option value="">Select Model</option>
                  {(CAR_MODEL_MAPPING[formData.brand] || []).map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>

            <div className="form-group">
              <label>Year</label>
              <select id="year" value={formData.year} onChange={handleInputChange}>
                {YEARS_LIST.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label>Seats</label>
              <div className="counter-row">
                <button className="counter-btn" onClick={() => updateSeats(-1)}>-</button>
                <span className="counter-value">{formData.seats}</span>
                <button className="counter-btn" onClick={() => updateSeats(1)}>+</button>
              </div>
            </div>

            <div className="form-group" style={{gridColumn: 'span 2'}}>
              <label>KM Driven: <span className="highlight-text">{formatNumber(formData.km_driven_input)} km</span></label>
              <input type="number" id="km_driven_input" value={formData.km_driven_input} onChange={handleInputChange} placeholder="e.g. 45000" />
            </div>

            <div className="form-group"><input type="number" id="mileage" onChange={handleInputChange} placeholder="Mileage (kmpl)"/></div>
            <div className="form-group"><input type="number" id="engine" onChange={handleInputChange} placeholder="Engine (CC)"/></div>
            
            <div className="form-group">
              <p id="what">Fuel Type</p>
              <select id="fuel_type" value={formData.fuel_type} onChange={handleInputChange}>
                <option value="Petrol">Petrol</option><option value="Diesel">Diesel</option><option value="CNG">CNG</option>
              </select>
            </div>
            <div className="form-group">
              <p id="what">Transmission</p>
              <select id="transmission_type" value={formData.transmission_type} onChange={handleInputChange}>
                <option value="Manual">Manual</option><option value="Automatic">Automatic</option>
              </select>
            </div>
          </div>

<button 
    className={`predict-button ${loading || isFormInvalid ? 'disabled' : ''}`} 
    onClick={getPrediction} 
    disabled={loading || isFormInvalid}
  >
    {loading ? 'Analyzing Market...' : isFormInvalid ? 'Fill All Fields' : 'Get Price Prediction'}
  </button>
        </div>

        <div className="card output-column">
          <h2>Valuation Summary</h2>
          {loading ? (
            <div className="skeleton-price"></div>
          ) : (
            <div className="price-box">
              <p className="price-label">Estimated Resale Price</p>
              <p className="price-text">₹ {prediction ? prediction.toLocaleString('en-IN') : '---'}</p>
            </div>
          )}

          {prediction && !loading && (
            <div className="insight-box">
              <p>💡 <strong>Insight:</strong> {marketInsight}</p>
            </div>
          )}
        </div>
      </div>

      {isBrandModalOpen && (
        <div className="modal-overlay" onClick={() => setIsBrandModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Select Vehicle Brand</h3>
              <button className="close-btn" onClick={() => setIsBrandModalOpen(false)}>&times;</button>
            </div>
            <div className="brand-scroller">
              {Object.keys(CAR_MODEL_MAPPING).sort().map(brand => (
              <div key={brand} className="brand-item" onClick={() => selectBrand(brand)}>
    <img 
      src={getLogoPath(brand)} 
      alt={brand} 
      className="brand-logo" 
      onError={(e) => {
        // Fallback to a generic placeholder if the SVG is missing
        e.target.src = "/src/assets/logos/default-car.svg"; 
      }}
    />
    <span>{brand}</span>
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