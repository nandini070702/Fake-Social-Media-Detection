// Load environment variables
require('dotenv').config();

// Import required packages
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');

const app = express();

// Middleware
app.use(express.json()); // Allows handling JSON data
app.use(cors()); // Enables communication between frontend and backend

// **🔹 Connect to MongoDB Atlas**
const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log("✅ MongoDB Connected");
    } catch (err) {
        console.error("❌ MongoDB Connection Error:", err.message);
        process.exit(1); // Stop the server if MongoDB connection fails
    }
};
connectDB(); // Call the function to connect to MongoDB

// **🔹 Import Routes**
const authRoutes = require('./routes/auth');
const reportRoutes = require('./routes/report'); // ✅ Import report routes correctly

// **🔹 Use Routes**
app.use('/auth', authRoutes);
app.use('/report', reportRoutes); // ✅ Ensure correct usage

// **🔹 Test API Route**
app.get('/', (req, res) => {
    res.send("✅ Backend is running!");
});

// **🔹 Start Server**
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
