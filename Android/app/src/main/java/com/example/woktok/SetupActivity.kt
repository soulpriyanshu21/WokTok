package com.example.woktok

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class SetupActivity : AppCompatActivity() {

    private val languages = listOf(
        "English", "Hindi", "Gujarati", "Marathi", "Telugu", "Punjabi",
        "Tamil", "Assamese", "Bengali", "Kannada", "Malayalam"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_setup)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 100)
        }

        val spinnerLang = findViewById<Spinner>(R.id.spinnerMyLang)
        val etRoom = findViewById<EditText>(R.id.etRoomCode)
        val btnJoin = findViewById<Button>(R.id.btnJoinRoom)

        spinnerLang.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, languages
        )

        // Prefill room if returning
        val savedRoom = RoomManager.getRoom(this)
        if (savedRoom.isNotEmpty()) etRoom.setText(savedRoom)

        btnJoin.setOnClickListener {
            val lang = spinnerLang.selectedItem.toString()
            val room = etRoom.text.toString().trim().uppercase()

            if (room.length < 3) {
                etRoom.error = "Enter at least 3 characters"
                return@setOnClickListener
            }

            RoomManager.saveSession(this, room, lang)

            Toast.makeText(this, "Joined room $room as $lang", Toast.LENGTH_SHORT).show()

            startActivity(Intent(this, ConversationActivity::class.java))
            finish()
        }
    }
}