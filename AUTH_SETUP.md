# Authentication System Setup Guide

I've successfully implemented a complete authentication system for your Fathy application with login, signup, logout, and account deletion features. Here's what has been added:

## 🔧 Backend Changes

### New Files Created:

1. **app/schemas/auth.py** - Authentication request/response models
2. **app/services/auth_service.py** - JWT token generation and password hashing
3. **app/routes/auth.py** - Authentication endpoints
4. **app/routes/dependencies.py** - Protected route dependencies

### Updated Files:

1. **app/db/models.py** - Added `User` model and `user_id` to `MemoryItem`
2. **app/main.py** - Registered auth router
3. **requirements.txt** - Added authentication dependencies

### New Backend Dependencies:

- `bcrypt==4.1.3` - Password hashing
- `passlib[bcrypt]==1.7.4` - Password utilities
- `python-jose[cryptography]==3.3.0` - JWT token handling
- `PyJWT==2.8.1` - JWT support
- `email-validator==2.1.1` - Email validation

### Authentication Endpoints:

- `POST /auth/register` - Create new account (email, username, password)
- `POST /auth/login` - Log in with email and password
- `POST /auth/logout` - Log out (invalidates token on client)
- `DELETE /auth/account` - Delete user account (requires authentication)
- `GET /auth/me` - Get current authenticated user info

## 🎨 Frontend Changes

### New Files Created:

1. **src/lib/auth-context.tsx** - Authentication context provider for state management
2. **src/components/LoginClient.tsx** - Login page component
3. **src/components/SignupClient.tsx** - Sign up page component
4. **src/app/auth/login/page.tsx** - Login page route
5. **src/app/auth/signup/page.tsx** - Sign up page route

### Updated Files:

1. **src/lib/api.ts** - Added auth API functions and token management
2. **src/app/layout.tsx** - Wrapped app with AuthProvider
3. **src/components/AppShell.tsx** - Added user menu with logout and delete account options

### Key Features:

- **Token Management**: Tokens stored in localStorage automatically
- **Auth Context**: Global user state management with `useAuth()` hook
- **Protected Routes**: Automatic token injection in authorized API calls
- **User Menu**: Displays logged-in user with dropdown for logout/delete account
- **Session Persistence**: Auto-login on page refresh if token exists

## 📋 Setup Instructions

### 1. Install Backend Dependencies:

```bash
cd "backend API"
python -m pip install -r requirements.txt
```

### 2. Create Database Tables:

Run your backend to create the new `users` table:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first run, the app will initialize the database. If you're using SQLAlchemy with automatic table creation, tables will be created and ready to use.

### 3. Frontend is Ready:

The frontend is already updated and will auto-detect the auth system.

## 🚀 How to Test

1. **Start Backend**:

   ```bash
   cd backend\ API
   uvicorn app.main:app --reload
   ```

2. **Start Frontend**:

   ```bash
   cd frontend\ API
   npm run dev
   ```

3. **Test Flow**:
   - Visit `http://localhost:3000/auth/signup`
   - Create a new account (email, username, password)
   - System automatically logs you in and redirects to chat
   - Click user menu (top right) to see logout/delete options
   - Log out or delete account as needed
   - Try `/auth/login` to log back in

## 🔐 Security Features

- **Password Hashing**: Uses bcrypt with salt for secure password storage
- **JWT Tokens**: 7-day expiration by default (configurable)
- **HTTP Bearer**: Authorization via Bearer tokens
- **CORS Enabled**: Already configured for your frontend
- **User Validation**: Email and username uniqueness enforced
- **Session Management**: Tokens stored securely in localStorage

## ⚙️ Configuration

### Modify Token Expiration:

Edit `app/services/auth_service.py`:

```python
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # Change this value
```

### Change Password Requirements:

Edit `app/schemas/auth.py` - adjust `min_length` and `max_length` in `UserCreate`

### Update CORS Origins:

Edit `.env` file - update `CORS_ORIGINS` for production

## 📝 Important Notes

- ✅ Tokens are automatically stored in localStorage
- ✅ Tokens are automatically sent with authenticated API requests
- ✅ Logout clears the token from localStorage
- ✅ Account deletion removes user and associated data
- ⚠️ The `deleteAccount` function is a hard delete - use confirmation dialogs
- ⚠️ Remember to update your `.env` file for production with proper `SECRET_KEY`

## 🔄 Next Steps (Optional)

1. **Add Email Verification**: Require email confirmation before account activation
2. **Add Password Reset**: Implement forgot password functionality
3. **Add Social Login**: Google, GitHub, etc.
4. **Add 2FA**: Two-factor authentication
5. **Add User Profiles**: Additional user information (avatar, bio, etc.)
6. **Link Memory to Users**: Ensure users can only access their own memory items

---

Your authentication system is now fully functional and ready to use! 🎉
