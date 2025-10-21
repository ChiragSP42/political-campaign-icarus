import React, { useState } from 'react';
import './QuestionnaireForm.css';

// REPLACE THIS WITH YOUR API GATEWAY URL FROM STEP 5.6
const API_ENDPOINT = 'https://sobrv4s9gc.execute-api.us-east-1.amazonaws.com/Initial_deployment';

const QuestionnaireForm = () => {
  const [formData, setFormData] = useState({
    fullName: '',
    office: '',
    district: '',
    militaryBackground: '',
    publicSafety: '',
    unionTies: '',
    smallBusiness: '',
    publicService: '',
    faithCommunity: '',
    campaignExperience: '',
    themeSong: '',
    debateReaction: '',
    coffeeShopIntro: '',
    leaderStyle: '',
    preferredEvent: '',
    decisionMaking: '',
    tagline: '',
    socialMediaVoice: '',
    opponentResponse: '',
    visionOfSuccess: '',
    campaignSymbol: '',
    campaignHeadline: '',
    requestDistrictAnalysis: false
  });

  const [submitStatus, setSubmitStatus] = useState({ 
    loading: false, 
    error: null, 
    success: null 
  });
  
  const [currentSection, setCurrentSection] = useState(0);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitStatus({ loading: true, error: null, success: null });

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {
        setSubmitStatus({
          loading: false,
          error: null,
          success: `Questionnaire submitted successfully! Submission ID: ${data.submissionId}`
        });
        
        // Reset form after 3 seconds
        setTimeout(() => {
          window.location.reload();
        }, 3000);
      } else {
        throw new Error(data.error || 'Submission failed');
      }

    } catch (error) {
      console.error('Error submitting questionnaire:', error);
      setSubmitStatus({
        loading: false,
        error: error.message || 'Failed to submit questionnaire. Please try again.',
        success: null
      });
    }
  };

  const renderBasicInfo = () => (
    <div className="form-section">
      <h2>Candidate Information</h2>
      
      <div className="form-group">
        <label htmlFor="fullName">Full Name *</label>
        <input
          type="text"
          id="fullName"
          name="fullName"
          value={formData.fullName}
          onChange={handleChange}
          required
          placeholder="Enter your full name"
        />
      </div>

      <div className="form-group">
        <label htmlFor="office">Office Running For *</label>
        <select
          id="office"
          name="office"
          value={formData.office}
          onChange={handleChange}
          required
        >
          <option value="">Select an office</option>
          <option value="house-delegates">Virginia House of Delegates</option>
          <option value="state-senate">Virginia State Senate</option>
          <option value="us-house">U.S. House of Representatives</option>
          <option value="us-senate">U.S. Senate</option>
          <option value="local-office">Local Office</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="district">District Running In *</label>
        <input
          type="text"
          id="district"
          name="district"
          value={formData.district}
          onChange={handleChange}
          required
          placeholder="e.g., District 10"
        />
      </div>
    </div>
  );

  const renderSection1 = () => (
    <div className="form-section">
      <h2>Section 1: Background & Profile</h2>
      <p className="section-description">
        These questions help us understand your credibility anchors.
      </p>

      <div className="form-group">
        <label htmlFor="militaryBackground">
          1. Do you have a military background?
        </label>
        <textarea
          id="militaryBackground"
          name="militaryBackground"
          value={formData.militaryBackground}
          onChange={handleChange}
          rows="3"
          placeholder="Please describe..."
        />
      </div>

      <div className="form-group">
        <label htmlFor="publicSafety">
          2. Have you served in law enforcement or public safety?
        </label>
        <textarea
          id="publicSafety"
          name="publicSafety"
          value={formData.publicSafety}
          onChange={handleChange}
          rows="3"
          placeholder="Please describe..."
        />
      </div>

      <div className="form-group">
        <label htmlFor="unionTies">
          3. Union household or labor ties?
        </label>
        <textarea
          id="unionTies"
          name="unionTies"
          value={formData.unionTies}
          onChange={handleChange}
          rows="3"
          placeholder="Please describe..."
        />
      </div>

      <div className="form-group">
        <label htmlFor="smallBusiness">
          4. Small business owner or entrepreneur?
        </label>
        <textarea
          id="smallBusiness"
          name="smallBusiness"
          value={formData.smallBusiness}
          onChange={handleChange}
          rows="3"
          placeholder="Please describe..."
        />
      </div>
    </div>
  );

  const renderSection2 = () => (
    <div className="form-section">
      <h2>Section 2: Communication Style</h2>
      
      <div className="form-group">
        <label>If your campaign had a theme song, what would it sound like?</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="themeSong"
              value="upbeat-pop"
              checked={formData.themeSong === 'upbeat-pop'}
              onChange={handleChange}
            />
            🎶 Upbeat pop anthem
          </label>
          <label>
            <input
              type="radio"
              name="themeSong"
              value="rock-hiphop"
              checked={formData.themeSong === 'rock-hiphop'}
              onChange={handleChange}
            />
            🎸 Rock/hip-hop banger
          </label>
          <label>
            <input
              type="radio"
              name="themeSong"
              value="folk-acoustic"
              checked={formData.themeSong === 'folk-acoustic'}
              onChange={handleChange}
            />
            🎻 Folk/acoustic ballad
          </label>
        </div>
      </div>

      <div className="form-group">
        <label>At a debate, you get a tough question. What do you do?</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="debateReaction"
              value="honest-answer"
              checked={formData.debateReaction === 'honest-answer'}
              onChange={handleChange}
            />
            Answer honestly
          </label>
          <label>
            <input
              type="radio"
              name="debateReaction"
              value="pivot-policy"
              checked={formData.debateReaction === 'pivot-policy'}
              onChange={handleChange}
            />
            Pivot to policy
          </label>
        </div>
      </div>

      <div className="form-group">
        <label>Campaign symbol?</label>
        <div className="radio-group">
          <label>
            <input
              type="radio"
              name="campaignSymbol"
              value="star"
              checked={formData.campaignSymbol === 'star'}
              onChange={handleChange}
            />
            ⭐ Star
          </label>
          <label>
            <input
              type="radio"
              name="campaignSymbol"
              value="bridge"
              checked={formData.campaignSymbol === 'bridge'}
              onChange={handleChange}
            />
            🌉 Bridge
          </label>
          <label>
            <input
              type="radio"
              name="campaignSymbol"
              value="flame"
              checked={formData.campaignSymbol === 'flame'}
              onChange={handleChange}
            />
            🔥 Flame
          </label>
        </div>
      </div>
    </div>
  );

  const renderSection3 = () => (
    <div className="form-section">
      <h2>Section 3: District Analysis</h2>
      
      <div className="form-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            name="requestDistrictAnalysis"
            checked={formData.requestDistrictAnalysis}
            onChange={handleChange}
          />
          ✅ Yes, analyze my district
        </label>
      </div>
    </div>
  );

  return (
    <div className="questionnaire-container">
      <div className="questionnaire-header">
        <h1>🗳️ Project Icarus Candidate Questionnaire</h1>
        <p>Help us understand your campaign profile</p>
      </div>

      <div className="progress-indicator">
        <div className={`progress-step ${currentSection >= 0 ? 'active' : ''}`}>
          Basic Info
        </div>
        <div className={`progress-step ${currentSection >= 1 ? 'active' : ''}`}>
          Background
        </div>
        <div className={`progress-step ${currentSection >= 2 ? 'active' : ''}`}>
          Style
        </div>
        <div className={`progress-step ${currentSection >= 3 ? 'active' : ''}`}>
          District
        </div>
      </div>

      <form onSubmit={handleSubmit} className="questionnaire-form">
        {currentSection === 0 && renderBasicInfo()}
        {currentSection === 1 && renderSection1()}
        {currentSection === 2 && renderSection2()}
        {currentSection === 3 && renderSection3()}

        <div className="form-navigation">
          {currentSection > 0 && (
            <button
              type="button"
              onClick={() => setCurrentSection(prev => prev - 1)}
              className="btn btn-secondary"
            >
              Previous
            </button>
          )}

          {currentSection < 3 && (
            <button
              type="button"
              onClick={() => setCurrentSection(prev => prev + 1)}
              className="btn btn-primary"
            >
              Next
            </button>
          )}

          {currentSection === 3 && (
            <button
              type="submit"
              disabled={submitStatus.loading}
              className="btn btn-submit"
            >
              {submitStatus.loading ? 'Submitting...' : 'Submit'}
            </button>
          )}
        </div>

        {submitStatus.error && (
          <div className="alert alert-error">
            {submitStatus.error}
          </div>
        )}

        {submitStatus.success && (
          <div className="alert alert-success">
            {submitStatus.success}
          </div>
        )}
      </form>
    </div>
  );
};

export default QuestionnaireForm;
