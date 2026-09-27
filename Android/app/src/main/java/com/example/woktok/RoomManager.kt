package com.example.woktok

import android.content.Context
import android.content.SharedPreferences

object RoomManager {

    private const val PREF = "woktok_prefs"
    private const val KEY_ROOM = "room_code"
    private const val KEY_LANG = "my_language"
    private const val KEY_USER_ID = "user_id"

    private fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE)

    fun saveSession(context: Context, room: String, language: String) {
        val userId = "user_" + System.currentTimeMillis()
        prefs(context).edit()
            .putString(KEY_ROOM, room)
            .putString(KEY_LANG, language)
            .putString(KEY_USER_ID, userId)
            .apply()
    }

    fun getRoom(context: Context): String =
        prefs(context).getString(KEY_ROOM, "") ?: ""

    fun getLanguage(context: Context): String =
        prefs(context).getString(KEY_LANG, "hindi") ?: "hindi"

    fun getUserId(context: Context): String =
        prefs(context).getString(KEY_USER_ID, "") ?: ""

    fun clear(context: Context) {
        prefs(context).edit().clear().apply()
    }
}