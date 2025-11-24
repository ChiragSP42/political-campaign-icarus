"""
Authentication management using Cognito
"""

import boto3
import logging
from typing import Optional, Tuple
from botocore.exceptions import ClientError
from utils.constants import (
    COGNITO_USER_POOL_ID,
    COGNITO_CLIENT_ID,
    COGNITO_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY
)

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages Cognito authentication"""
    
    def __init__(self):
        self.client = boto3.client(
            'cognito-idp',
            region_name=COGNITO_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        self.user_pool_id = COGNITO_USER_POOL_ID
        self.client_id = COGNITO_CLIENT_ID
    
    def sign_up(self, email: str, password: str) -> Tuple[bool, str]:
        """
        Sign up new user
        
        Returns:
            (success: bool, message: str)
        """
        try:
            response = self.client.sign_up(
                ClientId=self.client_id,
                Username=email,
                Password=password,
                UserAttributes=[
                    {
                        'Name': 'email',
                        'Value': email
                    }
                ]
            )
            
            logger.info(f"User {email} signed up successfully")
            return True, "Sign up successful! Check your email for verification code."
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'UsernameExistsException':
                return False, "Email already registered. Please sign in."
            elif error_code == 'InvalidPasswordException':
                return False, "Password must be 8+ characters with uppercase, lowercase, and numbers."
            elif error_code == 'InvalidParameterException':
                return False, "Invalid email format."
            else:
                return False, f"Sign up failed: {e.response['Error']['Message']}"
        
        except Exception as e:
            logger.error(f"Sign up error: {e}")
            return False, f"Error: {str(e)}"
    
    def confirm_sign_up(self, email: str, code: str) -> Tuple[bool, str]:
        """
        Confirm email verification code
        
        Returns:
            (success: bool, message: str)
        """
        try:
            self.client.confirm_sign_up(
                ClientId=self.client_id,
                Username=email,
                ConfirmationCode=code
            )
            
            logger.info(f"User {email} confirmed")
            return True, "Email confirmed! You can now sign in."
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'CodeMismatchException':
                return False, "Invalid verification code."
            elif error_code == 'UserNotFoundException':
                return False, "User not found."
            else:
                return False, f"Confirmation failed: {e.response['Error']['Message']}"
        
        except Exception as e:
            logger.error(f"Confirmation error: {e}")
            return False, f"Error: {str(e)}"
    
    def sign_in(self, email: str, password: str) -> Tuple[bool, str]:
        """
        Sign in user
        
        Returns:
            (success: bool, message: str)
        """
        try:
            response = self.client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': email,
                    'PASSWORD': password
                }
            )
            
            logger.info(f"User {email} signed in successfully")
            return True, email
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'NotAuthorizedException':
                return False, "Invalid email or password."
            elif error_code == 'UserNotFoundException':
                return False, "User not found. Please sign up first."
            elif error_code == 'UserNotConfirmedException':
                return False, "Please confirm your email first."
            else:
                return False, f"Sign in failed: {e.response['Error']['Message']}"
        
        except Exception as e:
            logger.error(f"Sign in error: {e}")
            return False, f"Error: {str(e)}"
    
    def change_password(self, email: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        try:
            # First sign in to get access token
            auth_response = self.client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': email,
                    'PASSWORD': old_password
                }
            )
            
            access_token = auth_response['AuthenticationResult']['AccessToken']
            
            # Now change password
            self.client.change_password(
                PreviousPassword=old_password,
                ProposedPassword=new_password,
                AccessToken=access_token
            )
            
            return True, "Password changed successfully."
        
        except Exception as e:
            logger.error(f"Password change error: {e}")
            return False, f"Error: {str(e)}"
