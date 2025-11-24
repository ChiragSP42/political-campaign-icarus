"""
API client for communicating with Lambda functions
"""

import requests
import json
import logging
from typing import Optional, Dict, Any, List
from utils.constants import API_ENDPOINT, API_TIMEOUT

logger = logging.getLogger(__name__)

class APIClient:
    """Handles API calls to Lambda functions"""
    
    def __init__(self, api_endpoint: str = API_ENDPOINT):
        self.api_endpoint = api_endpoint
        self.timeout = API_TIMEOUT
    
    def check_questionnaire_exists(self, email: str) -> Dict[str, Any]:
        """
        Check if user has completed questionnaire
        
        Args:
            email: User's email
            
        Returns:
            {exists: bool, email: str}
        """
        try:
            url = f"{self.api_endpoint}/check-questionnaire"
            params = {"email": email}
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            print(f"Response from check-questionnaire lambda: {response.json()}")
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking questionnaire: {e}")
            return {"exists": False, "error": str(e)}
    
    def save_questionnaire(self, email: Optional[str], answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save questionnaire answers
        
        Args:
            email: User's email
            answers: Questionnaire answers
            
        Returns:
            {success: bool, message: str}
        """
        try:
            url = f"{self.api_endpoint}/save-questionnaire"
            
            payload = {
                "email": email,
                "answers": answers
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error saving questionnaire: {e}")
            return {"success": False, "error": str(e)}
    
    def send_chat_message(
    self,
    email: Optional[str],
    message: str,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
        """
        Send chat message asynchronously and poll for response
        
        Returns:
            {success: bool, message: str}
        """
        try:
            # Step 1: Send async request
            url = f"{self.api_endpoint}/chat"
            payload = {
                "email": email,
                "query": message,
                "conversational_history": conversation_history
            }
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=payload, headers=headers, timeout=100)
            response.raise_for_status()
            result = response.json()
        
            return result
        
        except Exception as e:
            logger.error(f"Error sending requests.post() to chatbot: {e}")
            return {
                "status": "FAILED",
                "message": f"Error: {str(e)}"
            }
    
    def check_chatbot_response(self,
                               email: Optional[str]) -> Dict:
        """
        Check if chatbot has generated response and store in S3

        Args:
            email (Optional[str]): Email of user which is used for logical separation of files

        Returns:
            str: Returns the chatbot text
        """

        url = f"{self.api_endpoint}/check-response"
        payload={
            "email": email
        }
        headers = {"Content-Type": "application/json"}

        response = requests.get(url, params=payload, headers=headers, timeout=100)
        response.raise_for_status()
        result = response.json()

        return result
