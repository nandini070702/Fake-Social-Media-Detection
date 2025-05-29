const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const User = require('../models/user');
const authMiddleware = require('../middleware/authMiddleware');

const router = express.Router();

// **🔹 Protected Route**
router.get('/protected', authMiddleware, (req, res) => {
    res.json({ message: "You are accessing a protected route!", user: req.user });
});

// **🔹 Signup Route**
router.post('/signup', async (req, res) => {
    try {
        const { username, email, password } = req.body;

        // Check if all fields are provided
        if (!username || !email || !password) {
            return res.status(400).json({ message: "All fields are required!" });
        }

        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: "User already exists" });
        }

        // Hash password before saving
        const hashedPassword = await bcrypt.hash(password, 10);

        // Create new user
        const newUser = new User({ username, email, password: hashedPassword });
        await newUser.save();

        res.status(201).json({ message: "User registered successfully!" });
    } catch (error) {
        console.error("Signup Error:", error);
        res.status(500).json({ message: "Error signing up", error: error.message });
    }
});

// **🔹 Login Route**
router.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // Check if all fields are provided
        if (!email || !password) {
            return res.status(400).json({ message: "All fields are required!" });
        }

        // Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(404).json({ message: "User not found" });
        }

        // Compare entered password with stored hashed password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(400).json({ message: "Invalid credentials" });
        }

        // Ensure JWT_SECRET is set
        if (!process.env.JWT_SECRET) {
            return res.status(500).json({ message: "Server error: JWT secret is missing." });
        }

        // Generate JWT Token
        const token = jwt.sign(
            { userId: user._id, isAdmin: user.isAdmin },  // ✅ Now token includes isAdmin
            process.env.JWT_SECRET,
            { expiresIn: "1h" }
        );
        
        res.json({ message: "Login successful", token });
    } catch (error) {
        console.error("Login Error:", error);
        res.status(500).json({ message: "Error logging in", error: error.message });
    }
});

module.exports = router;

