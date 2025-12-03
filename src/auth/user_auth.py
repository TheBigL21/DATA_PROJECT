"""
USER AUTHENTICATION SYSTEM

Handles user registration, login, and region detection from phone numbers.
Stores user data securely with password hashing.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict
import phonenumbers
from phonenumbers import geocoder


class UserAuth:
    """User authentication and management system"""

    def __init__(self, users_db_path: str = "data/users.json"):
        """
        Initialize authentication system

        Args:
            users_db_path: Path to users database JSON file
        """
        self.users_db_path = Path(users_db_path)
        self.users_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        """Load users database from JSON"""
        if self.users_db_path.exists():
            with open(self.users_db_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_users(self):
        """Save users database to JSON"""
        with open(self.users_db_path, 'w') as f:
            json.dump(self.users, f, indent=2)

    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _extract_region(self, phone_number: str) -> tuple:
        """
        Extract region/country from phone number

        Args:
            phone_number: Phone number in international format (e.g., +1234567890)

        Returns:
            Tuple of (country_code, country_name, region)
            region is one of: 'North America', 'Europe', 'Asia', 'Latin America',
                               'Africa', 'Oceania', 'Other'
        """
        try:
            # Parse phone number
            parsed = phonenumbers.parse(phone_number, None)

            # Get country code
            country_code = parsed.country_code

            # Get country name
            country_name = geocoder.description_for_number(parsed, "en")

            # Map to broader regions
            region_mapping = {
                # North America
                1: "North America",  # US/Canada
                52: "North America",  # Mexico (can also be Latin America)

                # Europe
                33: "Europe",  # France
                34: "Europe",  # Spain
                39: "Europe",  # Italy
                41: "Europe",  # Switzerland
                43: "Europe",  # Austria
                44: "Europe",  # UK
                45: "Europe",  # Denmark
                46: "Europe",  # Sweden
                47: "Europe",  # Norway
                48: "Europe",  # Poland
                49: "Europe",  # Germany
                31: "Europe",  # Netherlands
                32: "Europe",  # Belgium
                351: "Europe",  # Portugal
                353: "Europe",  # Ireland
                358: "Europe",  # Finland
                420: "Europe",  # Czech Republic

                # Asia
                81: "Asia",  # Japan
                82: "Asia",  # South Korea
                86: "Asia",  # China
                91: "Asia",  # India
                65: "Asia",  # Singapore
                60: "Asia",  # Malaysia
                62: "Asia",  # Indonesia
                63: "Asia",  # Philippines
                66: "Asia",  # Thailand
                84: "Asia",  # Vietnam

                # Latin America
                54: "Latin America",  # Argentina
                55: "Latin America",  # Brazil
                56: "Latin America",  # Chile
                57: "Latin America",  # Colombia
                58: "Latin America",  # Venezuela

                # Oceania
                61: "Oceania",  # Australia
                64: "Oceania",  # New Zealand

                # Africa
                27: "Africa",  # South Africa
                20: "Africa",  # Egypt
                234: "Africa",  # Nigeria
            }

            region = region_mapping.get(country_code, "Other")

            return str(country_code), country_name, region

        except Exception as e:
            print(f"Warning: Could not parse phone number: {e}")
            return "Unknown", "Unknown", "Other"

    def signup(self, name: str, email: str, password: str, phone_number: str) -> tuple:
        """
        Register a new user

        Args:
            name: User's full name
            email: User's email (used as unique identifier)
            password: User's password (will be hashed)
            phone_number: User's phone number in international format

        Returns:
            Tuple of (success: bool, message: str, user_data: dict or None)
        """
        # Validate email doesn't already exist
        if email in self.users:
            return False, "Email already registered. Please login instead.", None

        # Extract region from phone number
        country_code, country_name, region = self._extract_region(phone_number)

        # Create user record
        user_data = {
            "name": name,
            "email": email,
            "password_hash": self._hash_password(password),
            "phone_number": phone_number,
            "country_code": country_code,
            "country": country_name,
            "region": region,
            "created_at": str(Path.cwd()),  # Simple timestamp alternative
            "preferences": {},
            "history": []
        }

        # Save user
        self.users[email] = user_data
        self._save_users()

        # Return safe user data (without password hash)
        safe_user_data = {k: v for k, v in user_data.items() if k != 'password_hash'}

        return True, f"Account created successfully! Region detected: {region}", safe_user_data

    def login(self, email: str, password: str) -> tuple:
        """
        Authenticate user login

        Args:
            email: User's email
            password: User's password

        Returns:
            Tuple of (success: bool, message: str, user_data: dict or None)
        """
        # Check if user exists
        if email not in self.users:
            return False, "Email not found. Please sign up first.", None

        # Verify password
        user = self.users[email]
        if user['password_hash'] != self._hash_password(password):
            return False, "Incorrect password.", None

        # Return safe user data
        safe_user_data = {k: v for k, v in user.items() if k != 'password_hash'}

        return True, f"Welcome back, {user['name']}!", safe_user_data

    def get_guest_profile(self) -> Dict:
        """
        Create a guest profile (for skip option)

        Returns:
            Guest user data with no region preferences
        """
        return {
            "name": "Guest",
            "email": "guest@local",
            "region": "Other",
            "country": "Unknown",
            "country_code": "Unknown",
            "preferences": {},
            "history": []
        }


def display_auth_menu() -> str:
    """
    Display authentication menu and get user choice

    Returns:
        Choice: '1' (Login), '2' (Signup), '3' (Skip)
    """
    import os
    os.system('clear' if os.name != 'nt' else 'cls')

    print("\n" + "="*60)
    print("  MOVIE FINDER - Welcome!")
    print("="*60 + "\n")

    print("Please choose an option:\n")
    print("  1. Login (existing user)")
    print("  2. Sign Up (new user)")
    print("  3. Skip (continue as guest)")

    while True:
        choice = input("\nSelect (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def handle_login(auth: UserAuth) -> Optional[Dict]:
    """
    Handle login flow

    Returns:
        User data if successful, None otherwise
    """
    print("\n" + "="*60)
    print("  LOGIN")
    print("="*60 + "\n")

    email = input("Email: ").strip()

    # Simple password input (in production, use getpass)
    import getpass
    password = getpass.getpass("Password: ")

    success, message, user_data = auth.login(email, password)
    print(f"\n{message}")

    if success:
        input("\nPress Enter to continue...")
        return user_data
    else:
        input("\nPress Enter to try again...")
        return None


def handle_signup(auth: UserAuth) -> Optional[Dict]:
    """
    Handle signup flow

    Returns:
        User data if successful, None otherwise
    """
    print("\n" + "="*60)
    print("  SIGN UP")
    print("="*60 + "\n")

    name = input("Full Name: ").strip()
    email = input("Email: ").strip()

    # Simple password input (in production, use getpass)
    import getpass
    password = getpass.getpass("Password: ")
    password_confirm = getpass.getpass("Confirm Password: ")

    if password != password_confirm:
        print("\nPasswords do not match!")
        input("Press Enter to try again...")
        return None

    print("\nPhone Number (international format, e.g., +1234567890):")
    print("This helps us recommend movies popular in your region.")
    phone_number = input("Phone: ").strip()

    success, message, user_data = auth.signup(name, email, password, phone_number)
    print(f"\n{message}")

    if success:
        input("\nPress Enter to continue...")
        return user_data
    else:
        input("\nPress Enter to try again...")
        return None


def authenticate() -> Dict:
    """
    Main authentication flow

    Returns:
        User data (authenticated user or guest)
    """
    auth = UserAuth()

    while True:
        choice = display_auth_menu()

        if choice == '1':  # Login
            user_data = handle_login(auth)
            if user_data:
                return user_data

        elif choice == '2':  # Signup
            user_data = handle_signup(auth)
            if user_data:
                return user_data

        elif choice == '3':  # Skip
            return auth.get_guest_profile()


if __name__ == '__main__':
    # Test authentication system
    user = authenticate()
    print(f"\nAuthentication complete!")
    print(f"User: {user['name']}")
    print(f"Region: {user['region']}")
