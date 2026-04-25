from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import MemoryItem, User
from app.db.session import get_db
from app.routes.dependencies import get_current_user
from app.schemas.auth import (
    DeleteAccountResponse,
    LogoutResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.services.auth_service import AuthService

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
):
    """Register a new user."""
    # Check if email already exists
    stmt = select(User).where(User.email == user_data.email)
    existing_email = db.execute(stmt).scalar_one_or_none()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Check if username already exists
    stmt = select(User).where(User.username == user_data.username)
    existing_username = db.execute(stmt).scalar_one_or_none()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken",
        )

    # Create new user
    hashed_password = AuthService.hash_password(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"New user registered: {new_user.email}")

    # Create access token
    access_token = AuthService.create_access_token(data={"sub": str(new_user.id)})

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(new_user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db),
):
    """Log in a user."""
    # Find user by email
    stmt = select(User).where(User.email == credentials.email)
    user = db.execute(stmt).scalar_one_or_none()

    if not user or not AuthService.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    logger.info(f"User logged in: {user.email}")

    # Create access token
    access_token = AuthService.create_access_token(data={"sub": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: User = Depends(get_current_user),
):
    """Log out a user (token invalidation on client-side)."""
    logger.info(f"User logged out: {current_user.email}")
    return LogoutResponse(message="Successfully logged out")


@router.delete("/account", response_model=DeleteAccountResponse)
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete the current user's account."""
    user_email = current_user.email
    
    # Delete user-owned memories first for compatibility with existing DBs
    db.execute(delete(MemoryItem).where(MemoryItem.user_id == current_user.id))

    # Delete user
    db.delete(current_user)
    db.commit()
    
    logger.info(f"User account deleted: {user_email}")
    return DeleteAccountResponse(message="Account successfully deleted")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get information about the current authenticated user."""
    return UserResponse.model_validate(current_user)
