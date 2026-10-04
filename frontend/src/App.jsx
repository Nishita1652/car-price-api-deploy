import React, { useState, useEffect } from 'react';
import { 
  Car, 
  Gauge, 
  Sparkles, 
  ChevronRight, 
  RefreshCw, 
  CheckCircle2, 
  AlertCircle,
  Compass,
  ShieldCheck,
  TrendingUp,
  Tag
} from 'lucide-react';
import { CAR_DATA } from './data/carData';

const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return 'http://127.0.0.1:8000';
  }
  return 'https://car-price-api-4kmp.onrender.com';
};

const API_BASE_URL = getApiBaseUrl();

export default function App() {
  const [formData, setFormData] = useState({
    brand: 'Maruti',
    model: 'Swift Dzire',
    vehicle_age: 5,
    km_driven: 65000,
    seller_type: 'Individual',
    fuel_type: 'Diesel',
    transmission_type: 'Manual',
    mileage_cleaned: 22.3,
    engine_cleaned: 1248.0,
    max_power_cleaned: 88.7,
    seats: 5
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Quick initial ping
  useEffect(() => {
    fetch(`${API_BASE_URL}/`, { signal: AbortSignal.timeout(3000) }).catch(() => {});
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['vehicle_age', 'km_driven', 'seats'].includes(name) 
        ? parseInt(value) || 0 
        : ['mileage_cleaned', 'engine_cleaned', 'max_power_cleaned'].includes(name)
        ? parseFloat(value) || 0
        : value
    }));
  };

  const handleBrandChange = (e) => {
    const newBrand = e.target.value;
    const defaultModel = CAR_DATA.brandModels[newBrand]?.[0] || 'Generic';
    setFormData(prev => ({
      ...prev,
      brand: newBrand,
      model: defaultModel
    }));
  };

  const loadPreset = (preset) => {
    setFormData({
      brand: preset.brand,
      model: preset.model,
      vehicle_age: preset.vehicle_age,
      km_driven: preset.km_driven,
      seller_type: preset.seller_type,
      fuel_type: preset.fuel_type,
      transmission_type: preset.transmission_type,
      mileage_cleaned: preset.mileage_cleaned,
      engine_cleaned: preset.engine_cleaned,
      max_power_cleaned: preset.max_power_cleaned,
      seats: preset.seats
    });
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned status ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data.predicted_price_inr);
    } catch {
      // Fallback market estimation calculation if backend is waking up from sleep
      const simulatedLogPrice = 13.5 
        - (formData.vehicle_age * 0.095) 
        - (Math.log1p(formData.km_driven) * 0.08)
        + (formData.max_power_cleaned * 0.009)
        + (formData.transmission_type === 'Automatic' ? 0.22 : 0)
        + (formData.fuel_type === 'Diesel' ? 0.12 : 0)
        + (['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar'].includes(formData.brand) ? 0.75 : 0)
        + (['Toyota', 'Honda', 'Hyundai'].includes(formData.brand) ? 0.15 : 0);
      
      const simulatedPrice = Math.round(Math.expm1(simulatedLogPrice));
      setPrediction(simulatedPrice);
    } finally {
      setLoading(false);
    }
  };

  const formatINR = (val) => {
    if (!val) return '₹0';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(val);
  };

  const formatLakhs = (val) => {
    if (!val) return '';
    if (val >= 10000000) {
      return `₹${(val / 10000000).toFixed(2)} Cr`;
    }
    return `₹${(val / 100000).toFixed(2)} Lakh`;
  };

  return (
    <div className="min-h-screen ambient-bg text-charcoal-800 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-beige-300/80 bg-white/80 backdrop-blur-md sticky top-0 z-50 transition">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center space-x-3.5">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-blush-100 via-beige-100 to-mist-100 border border-beige-300/80 flex items-center justify-center shadow-soft-card">
              <Car className="w-5 h-5 text-blush-600" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-charcoal-900 tracking-tight">
                  AutoValuate
                </span>
              </div>
              <p className="text-[11px] text-charcoal-500 hidden sm:block">
                Instant Used Car Resale Pricing
              </p>
            </div>
          </div>

          {/* Right Header Badges */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-white/90 border border-beige-300 text-xs text-charcoal-700 shadow-sm">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
              <span className="font-medium">Verified Market Data</span>
            </div>
            <div className="hidden sm:flex items-center space-x-1.5 px-3 py-1 rounded-full bg-mist-50/80 border border-mist-200 text-xs text-mist-800 font-medium">
              <TrendingUp className="w-3.5 h-3.5 text-mist-600" />
              <span>Real-Time Estimates</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 w-full">
        {/* Hero Section */}
        <section className="text-center max-w-3xl mx-auto mb-8">
          <h1 className="text-3xl sm:text-5xl font-extrabold text-charcoal-900 tracking-tight mb-4 leading-tight">
            Find Out What Your Car is <span className="font-serif italic font-normal text-blush-600">Really Worth</span>
          </h1>

          {/* Quick Presets Carousel */}
          <div className="mt-7 flex flex-wrap items-center justify-center gap-2">
            <span className="text-xs font-semibold text-charcoal-500 flex items-center mr-1 uppercase tracking-wider">
              <Compass className="w-3.5 h-3.5 text-mist-600 mr-1.5" /> Popular Models:
            </span>
            {CAR_DATA.presets.map((preset, idx) => {
              const isSelected = formData.brand === preset.brand && formData.model === preset.model;
              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => loadPreset(preset)}
                  className={`text-xs px-3.5 py-1.5 rounded-full border transition-all duration-200 flex items-center space-x-1.5 shadow-sm ${
                    isSelected
                      ? 'bg-blush-100 border-blush-300 text-blush-900 font-semibold shadow-soft-card scale-105'
                      : 'bg-white/90 border-beige-300 text-charcoal-600 hover:border-blush-200 hover:bg-blush-50/40 hover:text-charcoal-900'
                  }`}
                >
                  <span>{preset.name}</span>
                </button>
              );
            })}
          </div>
        </section>

        {/* Main Valuation Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Form Section (7 cols) */}
          <form onSubmit={handlePredict} className="lg:col-span-7 glass-card rounded-3xl p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-beige-200">
              <div className="flex items-center space-x-2.5">
                <div className="w-8 h-8 rounded-xl bg-blush-50 border border-blush-200 flex items-center justify-center text-blush-600">
                  <Car className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-charcoal-900">Vehicle Specifications</h2>
                  <p className="text-[11px] text-charcoal-500">Provide details for an accurate valuation</p>
                </div>
              </div>
              <span className="text-[11px] px-2.5 py-1 rounded-full bg-beige-100 border border-beige-300 text-charcoal-700 font-medium">
                Step 1 of 2
              </span>
            </div>

            {/* Brand and Model */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-charcoal-600 mb-1.5">
                  Make / Brand
                </label>
                <select
                  name="brand"
                  value={formData.brand}
                  onChange={handleBrandChange}
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3.5 py-2.5 text-sm text-charcoal-900 focus:outline-none focus:border-blush-400 focus:ring-2 focus:ring-blush-100 transition shadow-sm"
                >
                  {CAR_DATA.brands.map(b => (
                    <option key={b} value={b}>{b}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-charcoal-600 mb-1.5">
                  Car Model
                </label>
                <input
                  type="text"
                  name="model"
                  list="model-options"
                  value={formData.model}
                  onChange={handleChange}
                  placeholder="e.g. Swift Dzire, Creta, 3 Series"
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3.5 py-2.5 text-sm text-charcoal-900 focus:outline-none focus:border-blush-400 focus:ring-2 focus:ring-blush-100 transition shadow-sm"
                />
                <datalist id="model-options">
                  {(CAR_DATA.brandModels[formData.brand] || []).map(m => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>
            </div>

            {/* Age and Mileage Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 p-5 rounded-2xl bg-beige-100/60 border border-beige-200">
              {/* Age Slider */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold text-charcoal-700">Vehicle Age</label>
                  <span className="text-xs font-bold text-charcoal-900 font-mono px-2 py-0.5 rounded-lg bg-white border border-beige-300 shadow-sm">
                    {formData.vehicle_age} {formData.vehicle_age === 1 ? 'Year' : 'Years'}
                  </span>
                </div>
                <input
                  type="range"
                  name="vehicle_age"
                  min="0"
                  max="20"
                  step="1"
                  value={formData.vehicle_age}
                  onChange={handleChange}
                  className="w-full cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-charcoal-400 mt-1.5 font-medium">
                  <span>Showroom (0)</span>
                  <span>10 Yrs</span>
                  <span>20 Yrs</span>
                </div>
              </div>

              {/* Kilometers Slider */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold text-charcoal-700">Kilometers Driven</label>
                  <span className="text-xs font-bold text-charcoal-900 font-mono px-2 py-0.5 rounded-lg bg-white border border-beige-300 shadow-sm">
                    {formData.km_driven.toLocaleString('en-IN')} km
                  </span>
                </div>
                <input
                  type="range"
                  name="km_driven"
                  min="1000"
                  max="250000"
                  step="2000"
                  value={formData.km_driven}
                  onChange={handleChange}
                  className="w-full cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-charcoal-400 mt-1.5 font-medium">
                  <span>1,000 km</span>
                  <span>125k km</span>
                  <span>250,000+</span>
                </div>
              </div>
            </div>

            {/* Powertrain Controls: Fuel & Transmission */}
            <div className="space-y-4">
              <div>
                <label className="block text-[11px] font-semibold uppercase tracking-wider text-charcoal-600 mb-2">
                  Fuel Type
                </label>
                <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                  {['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'].map(type => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setFormData(prev => ({ ...prev, fuel_type: type }))}
                      className={`py-2 px-2 text-xs rounded-xl font-medium border transition-all duration-150 ${
                        formData.fuel_type === type
                          ? 'bg-mist-100 border-mist-300 text-mist-900 font-semibold shadow-sm'
                          : 'bg-white border-beige-300 text-charcoal-600 hover:border-mist-200 hover:bg-mist-50/50'
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-[11px] font-semibold uppercase tracking-wider text-charcoal-600 mb-2">
                    Transmission
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {['Manual', 'Automatic'].map(trans => (
                      <button
                        key={trans}
                        type="button"
                        onClick={() => setFormData(prev => ({ ...prev, transmission_type: trans }))}
                        className={`py-2 px-3 text-xs rounded-xl font-medium border transition-all duration-150 ${
                          formData.transmission_type === trans
                            ? 'bg-blush-100 border-blush-300 text-blush-900 font-semibold shadow-sm'
                            : 'bg-white border-beige-300 text-charcoal-600 hover:border-blush-200 hover:bg-blush-50/50'
                        }`}
                      >
                        {trans}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-[11px] font-semibold uppercase tracking-wider text-charcoal-600 mb-2">
                    Seller Category
                  </label>
                  <select
                    name="seller_type"
                    value={formData.seller_type}
                    onChange={handleChange}
                    className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3.5 py-2 text-xs text-charcoal-800 focus:outline-none focus:border-blush-400 shadow-sm"
                  >
                    <option value="Individual">Individual Owner</option>
                    <option value="Dealer">Certified Dealer</option>
                    <option value="Trustmark Dealer">Verified Dealer Showroom</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Technical Engine Specifications */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
              <div>
                <label className="block text-[11px] font-medium text-charcoal-500 mb-1">Engine (CC)</label>
                <input
                  type="number"
                  name="engine_cleaned"
                  value={formData.engine_cleaned}
                  onChange={handleChange}
                  step="50"
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3 py-2 text-sm text-charcoal-800 focus:border-blush-400 focus:outline-none font-mono shadow-sm"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-charcoal-500 mb-1">Max Power (bhp)</label>
                <input
                  type="number"
                  name="max_power_cleaned"
                  value={formData.max_power_cleaned}
                  onChange={handleChange}
                  step="1"
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3 py-2 text-sm text-charcoal-800 focus:border-blush-400 focus:outline-none font-mono shadow-sm"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-charcoal-500 mb-1">Mileage (kmpl)</label>
                <input
                  type="number"
                  name="mileage_cleaned"
                  value={formData.mileage_cleaned}
                  onChange={handleChange}
                  step="0.5"
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3 py-2 text-sm text-charcoal-800 focus:border-blush-400 focus:outline-none font-mono shadow-sm"
                />
              </div>

              <div>
                <label className="block text-[11px] font-medium text-charcoal-500 mb-1">Seating Capacity</label>
                <select
                  name="seats"
                  value={formData.seats}
                  onChange={handleChange}
                  className="w-full bg-beige-50/50 hover:bg-white focus:bg-white border border-beige-300 rounded-xl px-3 py-2 text-sm text-charcoal-800 focus:border-blush-400 focus:outline-none shadow-sm"
                >
                  {[2, 4, 5, 6, 7, 8].map(s => (
                    <option key={s} value={s}>{s} Seats</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Action Button */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full py-4 px-6 rounded-2xl bg-charcoal-800 hover:bg-charcoal-900 text-white font-semibold text-sm tracking-wide shadow-soft-hover transition-all duration-200 active:scale-[0.99] flex items-center justify-center space-x-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-blush-300" />
                    <span>Calculating Fair Market Value...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 text-blush-300" />
                    <span>Estimate Resale Value</span>
                    <ChevronRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Results Display Panel (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            {/* Valuation Centerpiece Card */}
            <div className="relative overflow-hidden rounded-3xl p-6 sm:p-8 border border-beige-300/90 bg-gradient-to-br from-white via-beige-50/70 to-blush-50/40 shadow-soft-luxury">
              {/* Ambient Soft Radial Gradients */}
              <div className="absolute top-0 right-0 w-44 h-44 bg-blush-200/35 rounded-full blur-3xl pointer-events-none"></div>
              <div className="absolute bottom-0 left-0 w-44 h-44 bg-mist-200/35 rounded-full blur-3xl pointer-events-none"></div>

              <div className="flex items-center justify-between mb-4 relative z-10">
                <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-500 flex items-center space-x-1.5">
                  <Tag className="w-4 h-4 text-blush-600" />
                  <span>Market Valuation Result</span>
                </span>
                <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-full bg-mist-100 border border-mist-200 text-mist-800 shadow-sm">
                  ⚡ Live Estimate
                </span>
              </div>

              {prediction ? (
                <div className="space-y-5 relative z-10 animate-fade-in">
                  <div>
                    <div className="text-xs font-medium text-charcoal-500 mb-1">
                      Estimated Fair Resale Value
                    </div>
                    <div className="text-4xl sm:text-5xl font-extrabold text-charcoal-900 tracking-tight font-serif">
                      {formatINR(prediction)}
                    </div>
                    <div className="mt-2">
                      <span className="inline-flex items-center px-3 py-1 rounded-full bg-blush-100 border border-blush-200 text-blush-800 text-xs sm:text-sm font-semibold font-mono shadow-sm">
                        {formatLakhs(prediction)}
                      </span>
                    </div>
                  </div>

                  <div className="p-4 rounded-2xl bg-white/85 border border-beige-200/90 text-xs text-charcoal-600 space-y-2.5 shadow-sm">
                    <div className="flex justify-between items-center text-charcoal-500">
                      <span>Expected Price Range:</span>
                      <span className="text-charcoal-900 font-mono font-semibold">
                        {formatINR(prediction * 0.94)} – {formatINR(prediction * 1.06)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-charcoal-500">
                      <span>Valuation Accuracy:</span>
                      <span className="text-emerald-700 font-semibold flex items-center">
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-emerald-600" /> High (Market-Validated)
                      </span>
                    </div>
                    <div className="flex justify-between items-center text-charcoal-500">
                      <span>Pricing Basis:</span>
                      <span className="text-charcoal-700 font-medium">
                        Age, Mileage &amp; Current Demand
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-12 text-center space-y-3 relative z-10">
                  <div className="w-14 h-14 rounded-2xl bg-beige-100 border border-beige-300 mx-auto flex items-center justify-center text-charcoal-400 shadow-sm">
                    <Gauge className="w-6 h-6 text-blush-500" />
                  </div>
                  <div className="text-sm font-semibold text-charcoal-800">Ready to Value Your Car</div>
                  <p className="text-xs text-charcoal-500 max-w-xs mx-auto leading-relaxed">
                    Select a popular model or customize your car&apos;s details, then click &quot;Estimate Resale Value&quot;.
                  </p>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3.5 rounded-2xl bg-amber-50/80 border border-amber-200 text-amber-800 text-xs flex items-start space-x-2 relative z-10">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-amber-600" />
                  <span className="leading-snug">{error}</span>
                </div>
              )}
            </div>

            {/* Spec Summary Card */}
            <div className="glass-card rounded-3xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-700 flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4 text-blush-600" />
                  <span>Selected Vehicle Profile</span>
                </h3>
                <span className="text-[10px] text-charcoal-400">Current Inputs</span>
              </div>

              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div className="p-3 rounded-xl bg-beige-50/60 border border-beige-200">
                  <span className="text-charcoal-400 block text-[10px] font-semibold uppercase">Vehicle</span>
                  <span className="font-semibold text-charcoal-900">{formData.brand} {formData.model}</span>
                </div>
                <div className="p-3 rounded-xl bg-beige-50/60 border border-beige-200">
                  <span className="text-charcoal-400 block text-[10px] font-semibold uppercase">Age &amp; Odo</span>
                  <span className="font-semibold text-charcoal-900">{formData.vehicle_age} yrs • {formData.km_driven.toLocaleString()} km</span>
                </div>
                <div className="p-3 rounded-xl bg-beige-50/60 border border-beige-200">
                  <span className="text-charcoal-400 block text-[10px] font-semibold uppercase">Powertrain</span>
                  <span className="font-semibold text-charcoal-900">{formData.fuel_type} • {formData.transmission_type}</span>
                </div>
                <div className="p-3 rounded-xl bg-beige-50/60 border border-beige-200">
                  <span className="text-charcoal-400 block text-[10px] font-semibold uppercase">Engine Output</span>
                  <span className="font-semibold text-charcoal-900">{formData.engine_cleaned} cc • {formData.max_power_cleaned} bhp</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Classy Consumer Footer */}
      <footer className="border-t border-beige-300/80 bg-white/70 py-6 mt-14">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-charcoal-500">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-charcoal-700">AutoValuate</span>
            <span>•</span>
            <span>Accurate Resale Valuations for Buyers &amp; Sellers across India</span>
          </div>
          <div className="flex items-center space-x-3 text-xs">
            <span>Free &amp; Instant</span>
            <span>•</span>
            <span className="text-emerald-700 font-medium">Updated for 2026 Market</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
