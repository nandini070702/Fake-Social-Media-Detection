const express = require('express');
const authMiddleware = require('../middleware/authMiddleware'); // Middleware to protect routes
const Report = require('../models/Report'); 

const router = express.Router();

// **🔹 Submit a Fake Profile Report (Protected Route)**
router.post('/submit', authMiddleware, async (req, res) => {
    try {
        const { reportedProfile, reason } = req.body;

        // Validate input fields
        if (!reportedProfile || !reason) {
            return res.status(400).json({ message: "All fields are required!" });
        }

        // Create and save the new report
        const newReport = new Report({
            userId: req.user.userId, // Extract user ID from JWT token
            reportedProfile,
            reason
        });

        await newReport.save();
        res.status(201).json({ message: "Report submitted successfully!", report: newReport });
    } catch (error) {
        console.error("❌ Report Submission Error:", error);
        res.status(500).json({ message: "Error submitting report", error: error.message });
    }
});

// **🔹 View Reports for Logged-in User (Protected Route)**
router.get('/my-reports', authMiddleware, async (req, res) => {
    try {
        // Fetch reports for the logged-in user
        const reports = await Report.find({ userId: req.user.userId });

        if (!reports || reports.length === 0) {
            return res.status(404).json({ message: "No reports found!" });
        }

        res.status(200).json({ reports });
    } catch (error) {
        console.error("❌ Fetch Reports Error:", error);
        res.status(500).json({ message: "Error fetching reports", error: error.message });
    }
});

// **🔹 View All Reports (Admin Access)**
router.get('/all-reports', authMiddleware, async (req, res) => {
    try {
        // Check if user is an admin (You can modify this condition)
        console.log("User Details:", req.user); // ✅ Debugging step

        if (!req.user.isAdmin) {
            return res.status(403).json({ message: "Access denied. Admins only!" });
        }
        

        // Fetch all reports
        const reports = await Report.find();

        if (!reports || reports.length === 0) {
            return res.status(404).json({ message: "No reports found!" });
        }

        res.status(200).json({ reports });
    } catch (error) {
        console.error("❌ Fetch All Reports Error:", error);
        res.status(500).json({ message: "Error fetching reports", error: error.message });
    }
});

// **🔹 Update Report Status (Admin Only)**
router.put('/update/:id', authMiddleware, async (req, res) => {
    try {
        const { status } = req.body;

        // Check if user is an admin
        if (!req.user.isAdmin) {
            return res.status(403).json({ message: "Access denied. Admins only!" });
        }

        // Validate status input
        if (!["Pending", "Reviewed", "Fake"].includes(status)) {
            return res.status(400).json({ message: "Invalid status value!" });
        }

        // Update the report
        const updatedReport = await Report.findByIdAndUpdate(
            req.params.id,
            { status },
            { new: true }
        );

        if (!updatedReport) {
            return res.status(404).json({ message: "Report not found!" });
        }

        res.status(200).json({ message: "Report status updated!", report: updatedReport });
    } catch (error) {
        console.error("❌ Update Report Error:", error);
        res.status(500).json({ message: "Error updating report", error: error.message });
    }
});

module.exports = router; // ✅ Ensure correct export
