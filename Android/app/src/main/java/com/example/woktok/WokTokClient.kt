package com.example.woktok

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.ResponseBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query
import java.io.File
import java.util.concurrent.TimeUnit

data class ReceiveResponse(
    val status: String,
    val id: String? = null,
    val audio_url: String? = null,
    val source_text: String? = null,
    val translated_text: String? = null,
    val from_lang: String? = null,
    val to_lang: String? = null
)

interface WokTokApi {

    @Multipart
    @POST("send")
    suspend fun sendAudio(
        @Part file: MultipartBody.Part,
        @Part("source_lang") sourceLang: okhttp3.RequestBody,
        @Part("target_lang") targetLang: okhttp3.RequestBody,
        @Part("room") room: okhttp3.RequestBody,
        @Part("user_id") userId: okhttp3.RequestBody
    ): Response<ResponseBody>

    @GET("receive")
    suspend fun receive(
        @Query("room") room: String,
        @Query("user_id") userId: String
    ): Response<ReceiveResponse>
}

object WokTokClient {

    // ⚠️ CHANGE THIS to your laptop's LAN IP
    // Find it with: ipconfig (look for IPv4 Address)
    // Example: "http://192.168.1.5:8000/"
    // For emulator testing: "http://10.0.2.2:8000/"
    private const val BASE_URL = "http://10.141.55.114/"

    private val okHttp = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        })
        .build()

    private val api: WokTokApi = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttp)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(WokTokApi::class.java)

    // Known language codes for the backend
    private val languageCodes = mapOf(
        "English" to "English",
        "Hindi" to "Hindi",
        "Gujarati" to "Gujarati",
        "Marathi" to "Marathi",
        "Telugu" to "Telugu",
        "Punjabi" to "Punjabi",
        "Tamil" to "Tamil",
        "Assamese" to "Assamese",
        "Bengali" to "Bengali",
        "Kannada" to "Kannada",
        "Malayalam" to "Malayalam"
    )

    fun normalizeLang(lang: String): String =
        languageCodes[lang.lowercase()] ?: "hindi"

    suspend fun sendAudio(
        context: Context,
        audioFile: File,
        sourceLang: String,
        targetLang: String,
        room: String,
        userId: String
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val audioPart = MultipartBody.Part.createFormData(
                "file",
                audioFile.name,
                audioFile.asRequestBody("audio/mp4".toMediaType())
            )
            val sourceBody = okhttp3.RequestBody.create("text/plain".toMediaType(), normalizeLang(sourceLang))
            val targetBody = okhttp3.RequestBody.create("text/plain".toMediaType(), normalizeLang(targetLang))
            val roomBody = okhttp3.RequestBody.create("text/plain".toMediaType(), room)
            val userBody = okhttp3.RequestBody.create("text/plain".toMediaType(), userId)

            val response = api.sendAudio(audioPart, sourceBody, targetBody, roomBody, userBody)
            response.isSuccessful
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    suspend fun receive(room: String, userId: String): ReceiveResponse? =
        withContext(Dispatchers.IO) {
            try {
                val response = api.receive(room, userId)
                if (!response.isSuccessful) return@withContext null
                response.body()
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }

    suspend fun downloadAudio(context: Context, url: String): File? =
        withContext(Dispatchers.IO) {
            try {
                val fullUrl = if (url.startsWith("http")) url else BASE_URL.trimEnd('/') + url
                val client = OkHttpClient()
                val request = okhttp3.Request.Builder().url(fullUrl).build()
                val response = client.newCall(request).execute()
                if (!response.isSuccessful) return@withContext null
                val body = response.body ?: return@withContext null
                val file = File(context.cacheDir, "received_${System.currentTimeMillis()}.wav")
                file.writeBytes(body.bytes())
                file
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }
}