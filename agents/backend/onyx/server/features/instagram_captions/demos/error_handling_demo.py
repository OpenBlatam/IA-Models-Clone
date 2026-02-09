from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
from typing import Dict, Any
from datetime import datetime, timezone
from utils.error_handling import (
from utils.validation import (
from typing import Any, List, Dict, Optional
import logging
Error Handling and Validation Demo

This script demonstrates the comprehensive error handling and validation system
for the Instagram Captions API with practical examples.
"


# Import our error handling and validation modules
    ErrorCode,
    InstagramCaptionsException,
    ValidationException,
    AuthenticationException,
    ResourceNotFoundException,
    RateLimitException,
    AIProcessingException,
    create_error_response,
    log_error,
    handle_api_errors,
    validate_string,
    validate_numeric,
    StringValidationConfig,
    NumericValidationConfig
)

    ContentType,
    ToneType,
    CONTENT_LIMITS,
    CaptionRequest,
    BatchCaptionRequest,
    CaptionResponse,
    validate_email,
    validate_url,
    validate_instagram_username,
    sanitize_html,
    validate_caption_content
)

# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_custom_exceptions():

    """demo_custom_exceptions function."""
Demonstrate custom exception handling.    print(n=*60)
    print(CUSTOM EXCEPTIONS DEMO")
    print(= 
    # Validation Exception
    try:
        raise ValidationException(
            message="Input validation failed",
            details={
                field_errors   [object Object]                   field             message": "Invalid email format",
                    type": "value_error"
                    }
                ]
            }
        )
    except ValidationException as e:
        print(f"Validation Exception: {e.message})
        print(f"Status Code: {e.status_code})
        print(f"Error Code: {e.error_code})
        print(f"Details: {json.dumps(e.details, indent=2)}")
    
    # Authentication Exception
    try:
        raise AuthenticationException(
            message="Invalid API key",
            details={api_key":***"}
        )
    except AuthenticationException as e:
        print(f"\nAuthentication Exception: {e.message})
        print(f"Status Code: {e.status_code}")
    
    # Resource Not Found Exception
    try:
        raise ResourceNotFoundException("caption", "12345   except ResourceNotFoundException as e:
        print(f"\nResource Not Found: {e.message})
        print(f"Status Code: {e.status_code}) 
    # Rate Limit Exception
    try:
        raise RateLimitException(retry_after=120  except RateLimitException as e:
        print(f"\nRate Limit: {e.message})
        print(fRetry After: {e.details.get('retry_after')} seconds")
    
    # AI Processing Exception
    try:
        raise AIProcessingException(
            message="AI model unavailable",
            details={"model": gpt-4", reason": "service_down"}
        )
    except AIProcessingException as e:
        print(f"\nAI Processing: {e.message})
        print(f"Details: {e.details})

def demo_error_response_creation():

    """demo_error_response_creation function."""
Demonstrate error response creation.    print(n="*60)
    print("ERROR RESPONSE CREATION DEMO")
    print(=
    
    # Create error response
    error_response = create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Multiple validation errors occurred,        details={
            field_errors                {"field":email", "message: Invalid format"},
         [object Object]field": "username, ssage": "Too short"}
            ]
        },
        request_id="demo-123,        path="/api/v1/captions",
        method=POST"
    )
    
    print("Error Response:")
    print(json.dumps(error_response, indent=2default=str))

def demo_input_validation():

    """demo_input_validation function."""
Demonstrate input validation utilities.    print(n="*60)
    print("INPUT VALIDATION DEMO")
    print(=
    
    # String validation
    string_config = StringValidationConfig(
        value="test@example.com",
        field_name="email",
        min_length=5,
        max_length=100
        pattern=r'^a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
    )
    
    result = validate_string(config=string_config)
    print(fString Validation Result: [object Object]json.dumps(result, indent=2)}")
    
    # Numeric validation
    numeric_config = NumericValidationConfig(
        value=42,
        field_name=age
        min_value=18,
        max_value=100        allow_zero=False,
        allow_negative=False
    )
    
    result = validate_numeric(config=numeric_config)
    print(f"\nNumeric Validation Result: [object Object]json.dumps(result, indent=2)}")
    
    # Email validation
    email_result = validate_email(email="user@example.com")
    print(f"\nEmail Validation: {json.dumps(email_result, indent=2)}")
    
    # URL validation
    url_result = validate_url(url="https://www.instagram.com/user")
    print(fundefinednURL Validation: {json.dumps(url_result, indent=2)}")
    
    # Instagram username validation
    username_result = validate_instagram_username(username="test_user_123print(f"\nUsername Validation: {json.dumps(username_result, indent=2)}")

def demo_pydantic_validation():
    
    """demo_pydantic_validation function."""
monstrate Pydantic v2 validation.    print(n="*60
    print("PYDANTIC VALIDATION DEMO")
    print(=)
    
    # Valid caption request
    try:
        valid_request = CaptionRequest(
            prompt="Create a professional caption for a business post",
            content_type=ContentType.POST,
            tone=ToneType.PROFESSIONAL,
            hashtags=["business,professional", "success"],
            max_length=500
        )
        print("Valid Request Created Successfully)
        print(f"Prompt: {valid_request.prompt})
        print(f"Content Type: {valid_request.content_type})
        print(f"Tone: {valid_request.tone})
        print(f"Hashtags: {valid_request.hashtags}")
    except Exception as e:
        print(fValidation Error: {e}")
    
    # Invalid caption request
    try:
        invalid_request = CaptionRequest(
            prompt=",  # Empty prompt
            content_type=ContentType.STORY,
            hashtags=["#invalid#hashtag", "tag"],
            max_length=3000  # Exceeds limit
        )
    except Exception as e:
        print(fnInvalid Request Error: {e}")
    
    # Batch request validation
    try:
        batch_request = BatchCaptionRequest(
            requests=[valid_request],
            batch_size=5
        )
        print(f"\nBatch Request Valid: [object Object]len(batch_request.requests)} requests")
    except Exception as e:
        print(f"Batch Validation Error: {e}")

def demo_content_validation():

    """demo_content_validation function."""
emonstrate content validation.    print(n="*60    print("CONTENT VALIDATION DEMO")
    print(=)
    
    # Valid caption content
    valid_caption = "This is a great post! #awesome #instagram #socialmedia"
    result = validate_caption_content(
        caption=valid_caption,
        content_type=ContentType.POST
    )
    print(f"Valid Caption Result: [object Object]json.dumps(result, indent=2)}")
    
    # Invalid caption (too long)
    long_caption = "x" *2500 result = validate_caption_content(
        caption=long_caption,
        content_type=ContentType.STORY
    )
    print(f"\nLong Caption Result: [object Object]json.dumps(result, indent=2)})    
    # Caption with too many hashtags
    hashtag_caption =Post content  + .join([f"#{i}" for i in range(35 result = validate_caption_content(
        caption=hashtag_caption,
        content_type=ContentType.POST
    )
    print(f"\nToo Many Hashtags Result: [object Object]json.dumps(result, indent=2)}")

def demo_html_sanitization():

    """demo_html_sanitization function."""
nstrate HTML sanitization.    print(n="*60   print(HTML SANITIZATION DEMO")
    print(=0    # HTML content with allowed and disallowed tags
    html_content = 
    <p>This is a paragraph with <b>bold</b> and <i>italic</i> text.</p>
    <script>alert('malicious')</script>
    <img src=x onerror=alert('xss') />
    <a href="javascript:alert(xss')>Click me</a>
    <strong>This is allowed</strong>
    "
    
    result = sanitize_html(html_content=html_content)
    print(Original HTML:")
    print(html_content)
    print(f"\nSanitized Result: [object Object]json.dumps(result, indent=2)}")

@handle_api_errors
async def demo_api_error_handling():
    
    """demo_api_error_handling function."""
onstrate API error handling decorator.    print(n="*60  print("API ERROR HANDLING DEMO")
    print(=60   
    # Simulate successful operation
    print("Simulating successful API call...")
    return {status": success", "data": caption generated}
    
    # Note: The decorator will catch any exceptions and convert them to HTTPException

async def demo_error_logging():

    """demo_error_logging function."""
Demonstrate error logging.    print(n="*60)
    print(ERROR LOGGING DEMO")
    print("=*60 
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except Exception as e:
        error_data = log_error(
            error=e,
            request_id="demo-log-123           context={"user_id": "user123eration": "caption_generation"}
        )
        print("Error Logged:")
        print(json.dumps(error_data, indent=2default=str))

def demo_content_type_limits():

    """demo_content_type_limits function."""
emonstrate content type limits.    print(n="*60    print("CONTENT TYPE LIMITS DEMO")
    print("=*60)  
    for content_type, limit in CONTENT_LIMITS.items():
        print(f"{content_type.value.upper()}: {limit} characters)  # Test different content types
    test_caption = "This is a test caption with some content."
    
    for content_type in ContentType:
        result = validate_caption_content(
            caption=test_caption,
            content_type=content_type
        )
        print(f"\n{content_type.value}:[object Object]result['is_valid']}")

# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

async def run_all_demos():
  
    """run_all_demos function."""
all error handling and validation demos."    print("INSTAGRAM CAPTIONS API - ERROR HANDLING & VALIDATION DEMO")
    print(=80    
    # Run all demos
    demo_custom_exceptions()
    demo_error_response_creation()
    demo_input_validation()
    demo_pydantic_validation()
    demo_content_validation()
    demo_html_sanitization()
    await demo_api_error_handling()
    await demo_error_logging()
    demo_content_type_limits()
    
    print(n="*80  print(DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def show_usage_examples():
   
    """show_usage_examples function."""
ow practical usage examples.    print(n="*60)
    print(USAGE EXAMPLES")
    print("="*60)
    
    print("""
1Creating a Caption Request:
   request = CaptionRequest(
       prompt="Create a professional caption",
       content_type=ContentType.POST,
       tone=ToneType.PROFESSIONAL,
       hashtags=["business", success"]
   )

2idating User Input:
   result = validate_email(email="user@example.com")
   if result["is_valid]:    email = result["email"]

3. Handling API Errors:
   @handle_api_errors
   async def generate_caption(request: CaptionRequest):
       
    """generate_caption function."""
# Your logic here
       pass

4Creating Custom Exceptions:
   raise ValidationException(
       message="Invalid input",
       details={"field": email",reason": format"}
   )

5 Sanitizing HTML:
   result = sanitize_html(html_content="<p>Safe</p><script>alert(xss')</script>)
   clean_html = result["sanitized]
   removed_tags = result["removed_tags"]
    """)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ ==__main__":
    # Run the demo
    asyncio.run(run_all_demos())
    
    # Show usage examples
    show_usage_examples() 