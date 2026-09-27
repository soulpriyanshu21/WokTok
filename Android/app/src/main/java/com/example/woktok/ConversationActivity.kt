package com.example.woktok

import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import android.view.animation.AnimationUtils
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ConversationActivity : AppCompatActivity() {

    private var player: MediaPlayer? = null
    private var recorder: AudioRecorder? = null
    private val handler = Handler(Looper.getMainLooper())
    private var polling = false
    private var isRecording = false

    private lateinit var tvHeader: TextView
    private lateinit var tvSubHeader: TextView
    private lateinit var btnMic: Button
    private lateinit var tvStatus: TextView
    private lateinit var historyLayout: LinearLayout

    private lateinit var room: String
    private lateinit var myLang: String
    private lateinit var userId: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_conversation)

        tvHeader = findViewById(R.id.tvHeader)
        tvSubHeader = findViewById(R.id.tvSubHeader)
        btnMic = findViewById(R.id.btnMic)
        tvStatus = findViewById(R.id.tvStatus)
        historyLayout = findViewById(R.id.historyLayout)

        room = RoomManager.getRoom(this)
        myLang = RoomManager.getLanguage(this)
        userId = RoomManager.getUserId(this)

        tvHeader.text = "Room: $room"
        tvSubHeader.text = "You speak: $myLang"

        btnMic.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!isRecording) startRecording()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (isRecording) stopAndSend()
                    true
                }
                else -> false
            }
        }

        startPolling()
    }

    private fun startRecording() {
        try {
            recorder = AudioRecorder(this)
            recorder?.startRecording()
            isRecording = true

            btnMic.text = "🎙️"
            btnMic.background = ContextCompat.getDrawable(this, R.drawable.bg_mic_button_recording)
            btnMic.startAnimation(AnimationUtils.loadAnimation(this, R.anim.pulse))

            tvStatus.text = "Listening..."
        } catch (e: Exception) {
            Toast.makeText(this, "Mic error: ${e.message}", Toast.LENGTH_SHORT).show()
            isRecording = false
        }
    }

    private fun stopAndSend() {
        isRecording = false

        btnMic.clearAnimation()
        btnMic.text = "🎤"
        btnMic.background = ContextCompat.getDrawable(this, R.drawable.bg_mic_button)

        val file = recorder?.stopRecording()

        if (file == null) {
            tvStatus.text = "Recording failed"
            return
        }

        tvStatus.text = "Sending..."

        lifecycleScope.launch {
            val targetLang = guessOtherLanguage(myLang)

            val success = WokTokClient.sendAudio(
                context = applicationContext,
                audioFile = file,
                sourceLang = myLang,
                targetLang = targetLang,
                room = room,
                userId = userId
            )

            if (success) {
                tvStatus.text = "Sent to $targetLang listeners"
                Toast.makeText(this@ConversationActivity, "Message sent!", Toast.LENGTH_SHORT).show()
            } else {
                tvStatus.text = "Send failed. Is server running?"
            }
        }
    }

    private fun guessOtherLanguage(mine: String): String {
        return when (mine.lowercase()) {
            "hindi" -> "tamil"
            "tamil" -> "hindi"
            "english" -> "hindi"
            "bengali" -> "hindi"
            "marathi" -> "hindi"
            "telugu" -> "hindi"
            "kannada" -> "hindi"
            "malayalam" -> "hindi"
            "gujarati" -> "hindi"
            "punjabi" -> "hindi"
            "assamese" -> "hindi"
            else -> "hindi"
        }
    }

    private fun startPolling() {
        polling = true
        pollLoop()
    }

    private fun pollLoop() {
        if (!polling) return

        lifecycleScope.launch {
            val response = WokTokClient.receive(room, userId)
            if (response != null && response.status == "ok" && response.audio_url != null) {
                handleIncoming(response)
            }
        }

        handler.postDelayed({ pollLoop() }, 2000)
    }

    private suspend fun handleIncoming(response: ReceiveResponse) {
        val audioFile = WokTokClient.downloadAudio(this, response.audio_url!!)
        if (audioFile == null) {
            runOnUiThread { tvStatus.text = "Failed to download audio" }
            return
        }

        runOnUiThread {
            tvStatus.text = "Playing incoming..."
        }

        player?.release()
        player = MediaPlayer().apply {
            setDataSource(audioFile.absolutePath)
            setOnCompletionListener {
                runOnUiThread { tvStatus.text = "Ready" }
            }
            prepare()
            start()
        }

        runOnUiThread {
            addHistoryCard(
                from = response.from_lang ?: "?",
                text = response.translated_text ?: response.source_text ?: "",
                audioFile = audioFile,
                isIncoming = true
            )
        }
    }

    private fun addHistoryCard(from: String, text: String, audioFile: File, isIncoming: Boolean) {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 28, 28, 28)
            setBackgroundColor(0xFF0F141C.toInt())
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(0, 10, 0, 10)
            }
        }

        val tvFrom = TextView(this).apply {
            this.text = if (isIncoming) "📥 From $from" else "📤 You sent"
            setTextColor(0xFF58A6FF.toInt())
            textSize = 12f
        }

        val tvText = TextView(this).apply {
            this.text = text
            setTextColor(0xFFE6EDF3.toInt())
            textSize = 16f
            setPadding(0, 10, 0, 10)
        }

        val tvTime = TextView(this).apply {
            this.text = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
            setTextColor(0xFF6E7681.toInt())
            textSize = 11f
        }

        val btnReplay = Button(this).apply {
            this.text = "▶  Replay"
            textSize = 13f
            setTextColor(0xFF58A6FF.toInt())
            setBackgroundColor(0xFF05070D.toInt())
            setOnClickListener {
                player?.release()
                player = MediaPlayer().apply {
                    setDataSource(audioFile.absolutePath)
                    prepare()
                    start()
                }
            }
        }

        card.addView(tvFrom)
        card.addView(tvText)
        card.addView(tvTime)
        card.addView(btnReplay)

        historyLayout.addView(card, 0)
    }

    override fun onDestroy() {
        super.onDestroy()
        polling = false
        player?.release()
        player = null
    }
}